"""
Stress testing for RecallBricksMemory - Enterprise Grade Testing

Tests for:
- Thread safety and race conditions
- Concurrent user operations
- Memory leaks under sustained load
- Edge cases at scale
- Connection pooling behavior
- Error recovery
- Rate limiting scenarios

Run with: python tests/stress_test.py
"""

import unittest
import threading
import time
import random
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
from collections import defaultdict
from recallbricks_langchain import RecallBricksMemory


class StressTest(unittest.TestCase):
    """Enterprise-grade stress testing for RecallBricksMemory."""

    def setUp(self):
        """Set up test environment."""
        self.api_key = "test-api-key"
        self.results = defaultdict(list)
        self.errors = []
        self.lock = threading.Lock()

    def test_concurrent_writes_same_user(self):
        """Test multiple threads writing for the same user simultaneously."""
        print("\n" + "="*60)
        print("TEST: Concurrent Writes - Same User")
        print("="*60)

        user_id = "concurrent-test-user"
        num_threads = 50
        operations_per_thread = 20

        with patch('recallbricks_langchain.memory.RecallBricks') as mock_rb_class:
            mock_rb = Mock()
            mock_rb_class.return_value = mock_rb
            mock_rb.create_memory.return_value = {"id": "test", "status": "success"}

            memory = RecallBricksMemory(api_key=self.api_key, user_id=user_id)

            def worker(thread_id):
                """Worker thread that performs multiple saves."""
                for i in range(operations_per_thread):
                    try:
                        inputs = {"input": f"Thread {thread_id} message {i}"}
                        outputs = {"output": f"Response {i}"}
                        memory.save_context(inputs, outputs)
                        with self.lock:
                            self.results['concurrent_writes'].append(('success', thread_id, i))
                    except Exception as e:
                        with self.lock:
                            self.errors.append(('concurrent_write', thread_id, str(e)))

            # Execute concurrent writes
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker, i) for i in range(num_threads)]
                for future in as_completed(futures):
                    future.result()

            elapsed = time.time() - start_time

            # Verify all operations completed
            expected_ops = num_threads * operations_per_thread
            actual_ops = len(self.results['concurrent_writes'])

            print(f"Expected operations: {expected_ops}")
            print(f"Actual operations: {actual_ops}")
            print(f"Errors: {len(self.errors)}")
            print(f"Time elapsed: {elapsed:.2f}s")
            print(f"Operations/sec: {actual_ops/elapsed:.2f}")
            print(f"Total API calls: {mock_rb.create_memory.call_count}")

            self.assertEqual(actual_ops, expected_ops, "Some operations failed")
            self.assertEqual(len(self.errors), 0, f"Errors occurred: {self.errors}")

    def test_concurrent_reads_multiple_users(self):
        """Test concurrent reads across different users."""
        print("\n" + "="*60)
        print("TEST: Concurrent Reads - Multiple Users")
        print("="*60)

        num_users = 100
        operations_per_user = 10

        with patch('recallbricks_langchain.memory.RecallBricks') as mock_rb_class:
            mock_rb = Mock()
            mock_rb_class.return_value = mock_rb

            # Mock search results
            def mock_search(*args, **kwargs):
                time.sleep(random.uniform(0.001, 0.01))  # Simulate API latency
                result = Mock()
                result.text = "Human: test\nAI: response"
                return [result]

            mock_rb.search.side_effect = mock_search

            def worker(user_id):
                """Worker for each user."""
                memory = RecallBricksMemory(
                    api_key=self.api_key,
                    user_id=f"user-{user_id}"
                )

                for i in range(operations_per_user):
                    try:
                        inputs = {"input": f"Query {i}"}
                        result = memory.load_memory_variables(inputs)
                        with self.lock:
                            self.results['concurrent_reads'].append(('success', user_id))
                    except Exception as e:
                        with self.lock:
                            self.errors.append(('concurrent_read', user_id, str(e)))

            start_time = time.time()
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(worker, i) for i in range(num_users)]
                for future in as_completed(futures):
                    future.result()

            elapsed = time.time() - start_time

            expected_ops = num_users * operations_per_user
            actual_ops = len(self.results['concurrent_reads'])

            print(f"Users: {num_users}")
            print(f"Expected operations: {expected_ops}")
            print(f"Actual operations: {actual_ops}")
            print(f"Errors: {len(self.errors)}")
            print(f"Time elapsed: {elapsed:.2f}s")
            print(f"Operations/sec: {actual_ops/elapsed:.2f}")

            self.assertEqual(actual_ops, expected_ops)
            self.assertEqual(len(self.errors), 0)

    def test_memory_leak_detection(self):
        """Test for memory leaks under sustained load."""
        print("\n" + "="*60)
        print("TEST: Memory Leak Detection")
        print("="*60)

        process = psutil.Process()
        iterations = 1000
        samples = 10

        with patch('recallbricks_langchain.memory.RecallBricks') as mock_rb_class:
            mock_rb = Mock()
            mock_rb_class.return_value = mock_rb
            mock_rb.search.return_value = []
            mock_rb.create_memory.return_value = {"id": "test"}

            memory_samples = []

            for sample in range(samples):
                # Force garbage collection before measurement
                gc.collect()

                # Measure memory
                mem_before = process.memory_info().rss / 1024 / 1024

                # Perform operations
                for i in range(iterations):
                    memory = RecallBricksMemory(api_key=self.api_key, user_id=f"leak-test-{i}")

                    inputs = {"input": f"Test {i}" * 100}  # Large payload
                    outputs = {"output": f"Response {i}" * 100}

                    memory.save_context(inputs, outputs)
                    memory.load_memory_variables(inputs)

                    # Clean up
                    del memory

                gc.collect()
                mem_after = process.memory_info().rss / 1024 / 1024
                memory_samples.append(mem_after - mem_before)

                print(f"Sample {sample + 1}/{samples}: {mem_after - mem_before:.2f} MB delta")

            # Check if memory is steadily increasing (leak indicator)
            avg_increase = sum(memory_samples) / len(memory_samples)
            trend = sum(memory_samples[i] > memory_samples[i-1]
                       for i in range(1, len(memory_samples)))

            print(f"\nAverage memory increase per sample: {avg_increase:.2f} MB")
            print(f"Upward trend count: {trend}/{samples-1}")

            # Alert if consistent memory growth
            if trend > (samples * 0.7):  # 70% of samples showing growth
                print("⚠️  WARNING: Potential memory leak detected")
            else:
                print("✓ No significant memory leak detected")

            # Memory should stabilize (not grow indefinitely)
            self.assertLess(trend, samples * 0.8, "Potential memory leak detected")

    def test_edge_cases_at_scale(self):
        """Test edge cases with large scale."""
        print("\n" + "="*60)
        print("TEST: Edge Cases at Scale")
        print("="*60)

        with patch('recallbricks_langchain.memory.RecallBricks') as mock_rb_class:
            mock_rb = Mock()
            mock_rb_class.return_value = mock_rb
            mock_rb.search.return_value = []
            mock_rb.create_memory.return_value = {"id": "test"}

            test_cases = [
                ("Empty strings", {"input": ""}, {"output": ""}),
                ("Very long text", {"input": "x" * 100000}, {"output": "y" * 100000}),
                ("Unicode/Emoji", {"input": "🚀" * 1000}, {"output": "✅" * 1000}),
                ("Special chars", {"input": "\n\r\t" * 100}, {"output": "!@#$%^&*()" * 100}),
                ("Nested quotes", {"input": '"\'"\'"' * 100}, {"output": '"\'"\'"\'' * 100}),
                ("SQL injection", {"input": "'; DROP TABLE--"}, {"output": "safe"}),
                ("XSS attempt", {"input": "<script>alert('xss')</script>"}, {"output": "safe"}),
                ("Null bytes", {"input": "test\x00null"}, {"output": "response"}),
            ]

            for test_name, inputs, outputs in test_cases:
                print(f"\nTesting: {test_name}")
                try:
                    memory = RecallBricksMemory(api_key=self.api_key)
                    memory.save_context(inputs, outputs)
                    result = memory.load_memory_variables(inputs)
                    print(f"  ✓ {test_name} passed")
                except Exception as e:
                    print(f"  ✗ {test_name} failed: {e}")
                    self.errors.append((test_name, str(e)))

            self.assertEqual(len(self.errors), 0, f"Edge case failures: {self.errors}")

    def test_rapid_user_creation(self):
        """Test rapid creation of many users (connection pool test)."""
        print("\n" + "="*60)
        print("TEST: Rapid User Creation (Connection Pooling)")
        print("="*60)

        num_users = 1000

        with patch('recallbricks_langchain.memory.RecallBricks') as mock_rb_class:
            instances_created = []

            def track_rb_init(api_key):
                instance = Mock()
                instance.search.return_value = []
                instance.create_memory.return_value = {"id": "test"}
                instances_created.append(instance)
                return instance

            mock_rb_class.side_effect = track_rb_init

            start_time = time.time()

            # Rapidly create many memory instances
            memories = []
            for i in range(num_users):
                memory = RecallBricksMemory(
                    api_key=self.api_key,
                    user_id=f"rapid-user-{i}"
                )
                memories.append(memory)

            elapsed = time.time() - start_time

            print(f"Created {num_users} users in {elapsed:.2f}s")
            print(f"Users/sec: {num_users/elapsed:.2f}")
            print(f"RecallBricks instances created: {len(instances_created)}")

            # Each memory instance creates its own RB client currently
            # In production, you might want connection pooling
            self.assertEqual(len(instances_created), num_users)

            print("\n⚠️  NOTE: Each memory instance creates its own RecallBricks client")
            print("   Consider implementing connection pooling for production")

    def test_error_handling_at_scale(self):
        """Test error handling when API fails."""
        print("\n" + "="*60)
        print("TEST: Error Handling at Scale")
        print("="*60)

        num_operations = 100
        failure_rate = 0.3  # 30% failure rate

        with patch('recallbricks_langchain.memory.RecallBricks') as mock_rb_class:
            mock_rb = Mock()
            mock_rb_class.return_value = mock_rb

            call_count = [0]

            def flaky_search(*args, **kwargs):
                call_count[0] += 1
                if random.random() < failure_rate:
                    raise Exception("API Error: Rate limit exceeded")
                return []

            mock_rb.search.side_effect = flaky_search

            memory = RecallBricksMemory(api_key=self.api_key)

            successes = 0
            failures = 0

            for i in range(num_operations):
                try:
                    inputs = {"input": f"Query {i}"}
                    memory.load_memory_variables(inputs)
                    successes += 1
                except Exception:
                    failures += 1

            print(f"Total operations: {num_operations}")
            print(f"Successes: {successes}")
            print(f"Failures: {failures}")
            print(f"Actual failure rate: {failures/num_operations:.1%}")

            print("\n⚠️  WARNING: No retry logic implemented")
            print("   Add exponential backoff and retry logic for production")

            # Errors currently bubble up (no retry logic)
            self.assertGreater(failures, 0, "Should have some failures with flaky API")


def run_stress_tests():
    """Run all stress tests with reporting."""
    print("\n" + "="*60)
    print("RECALLBRICKS LANGCHAIN - ENTERPRISE STRESS TESTING")
    print("="*60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python process: {psutil.Process().pid}")

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    print(f"Initial memory: {mem_before:.2f} MB")
    print("="*60)

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(StressTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Final report
    mem_after = process.memory_info().rss / 1024 / 1024

    print("\n" + "="*60)
    print("STRESS TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Final memory: {mem_after:.2f} MB")
    print(f"Memory delta: {mem_after - mem_before:.2f} MB")
    print("="*60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_stress_tests()
    exit(0 if success else 1)
