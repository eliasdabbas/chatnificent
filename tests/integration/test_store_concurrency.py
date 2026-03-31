"""Concurrency tests for stateful store implementations.

Validates thread safety of InMemory, File, and SQLite stores under concurrent
access using real threads (not sequential "concurrent" simulations).
"""

import concurrent.futures

import pytest
from chatnificent.models import Conversation


class TestStoreConcurrency:
    """Thread safety validation across all store implementations."""

    @pytest.fixture(params=["InMemory", "File", "SQLite"])
    def store(self, request, tmp_path):
        from chatnificent import store as store_mod

        if request.param == "InMemory":
            return store_mod.InMemory()
        elif request.param == "File":
            return store_mod.File(str(tmp_path / "file_store"))
        elif request.param == "SQLite":
            return store_mod.SQLite(str(tmp_path / "test.db"))

    def test_concurrent_save_different_conversations(self, store):
        """10 threads saving different conversations — all must persist."""
        num_threads = 10

        def save_convo(i):
            convo = Conversation(
                id=f"convo_{i}",
                messages=[{"role": "user", "content": f"msg {i}"}],
            )
            store.save_conversation("user1", convo)
            return f"convo_{i}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(save_convo, i) for i in range(num_threads)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == num_threads

        saved_ids = store.list_conversations("user1")
        for i in range(num_threads):
            assert f"convo_{i}" in saved_ids

        for i in range(num_threads):
            loaded = store.load_conversation("user1", f"convo_{i}")
            assert loaded is not None
            assert loaded.messages[0]["content"] == f"msg {i}"

    def test_concurrent_save_same_conversation(self, store):
        """10 threads writing the same conversation — no corruption."""
        num_threads = 10

        def save_version(i):
            convo = Conversation(
                id="shared",
                messages=[{"role": "user", "content": f"version {i}"}],
            )
            store.save_conversation("user1", convo)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(save_version, i) for i in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        loaded = store.load_conversation("user1", "shared")
        assert loaded is not None
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"].startswith("version ")

    def test_concurrent_save_and_list(self, store):
        """Concurrent saves + list_conversations — list never returns corrupt data."""
        num_writers = 10
        num_readers = 5

        def writer(i):
            convo = Conversation(
                id=f"convo_{i}",
                messages=[{"role": "user", "content": f"msg {i}"}],
            )
            store.save_conversation("user1", convo)

        def reader():
            ids = store.list_conversations("user1")
            assert isinstance(ids, list)
            for cid in ids:
                assert isinstance(cid, str)
            return ids

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_writers + num_readers
        ) as pool:
            write_futures = [pool.submit(writer, i) for i in range(num_writers)]
            read_futures = [pool.submit(reader) for _ in range(num_readers)]

            for f in concurrent.futures.as_completed(write_futures + read_futures):
                f.result()

        final_ids = store.list_conversations("user1")
        assert len(final_ids) == num_writers

    def test_concurrent_file_append(self, store):
        """Concurrent save_file with append=True — all appends must be present."""
        convo = Conversation(
            id="append_test",
            messages=[{"role": "user", "content": "hello"}],
        )
        store.save_conversation("user1", convo)

        num_threads = 10

        def appender(i):
            line = f'{{"turn": {i}}}\n'
            store.save_file(
                "user1",
                "append_test",
                "log.jsonl",
                line.encode("utf-8"),
                append=True,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(appender, i) for i in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        data = store.load_file("user1", "append_test", "log.jsonl")
        assert data is not None

        lines = [line for line in data.decode("utf-8").splitlines() if line.strip()]
        assert len(lines) == num_threads

    def test_concurrent_multi_user_isolation(self, store):
        """Concurrent operations across different users — no cross-contamination."""
        num_users = 5
        convos_per_user = 3

        def save_user_convos(user_idx):
            user_id = f"user_{user_idx}"
            for j in range(convos_per_user):
                convo = Conversation(
                    id=f"c_{j}",
                    messages=[{"role": "user", "content": f"user{user_idx} msg{j}"}],
                )
                store.save_conversation(user_id, convo)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as pool:
            futures = [pool.submit(save_user_convos, i) for i in range(num_users)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        for i in range(num_users):
            user_id = f"user_{i}"
            ids = store.list_conversations(user_id)
            assert len(ids) == convos_per_user
            for j in range(convos_per_user):
                loaded = store.load_conversation(user_id, f"c_{j}")
                assert loaded is not None
                assert loaded.messages[0]["content"] == f"user{i} msg{j}"
