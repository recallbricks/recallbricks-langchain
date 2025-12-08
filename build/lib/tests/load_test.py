"""
Load testing script for RecallBricksMemory using Locust.

This simulates thousands of concurrent users to test:
- Throughput and latency
- Connection pooling
- Rate limiting
- Error handling under load
- Memory usage patterns

Run with: locust -f tests/load_test.py --headless -u 1000 -r 100 --run-time 5m
"""

from locust import User, task, between, events
from recallbricks_langchain import RecallBricksMemory
from unittest.mock import Mock, patch
import time
import random
import psutil
import os
from datetime import datetime


class RecallBricksUser(User):
    """Simulates a user interacting with RecallBricksMemory."""

    wait_time = between(0.1, 2)  # Simulate realistic user behavior

    def on_start(self):
        """Initialize user session."""
        self.user_id = f"load-test-user-{self.environment.runner.user_count}-{random.randint(1000, 9999)}"

        # Mock RecallBricks to avoid actual API calls during load testing
        # In production, you'd test against a staging environment
        with patch('recallbricks_langchain.memory.RecallBricks') as mock_rb_class:
            self.mock_rb = Mock()
            mock_rb_class.return_value = self.mock_rb

            # Configure mock responses
            self.mock_rb.search.return_value = self._mock_search_results()
            self.mock_rb.create_memory.return_value = {"id": "mock-id", "status": "success"}

            self.memory = RecallBricksMemory(
                api_key="test-key",
                user_id=self.user_id,
                limit=20
            )

    def _mock_search_results(self):
        """Generate mock search results."""
        results = []
        for i in range(random.randint(0, 10)):
            result = Mock()
            result.text = f"Human: Test query {i}\nAI: Test response {i}"
            result.relevance = random.uniform(0.5, 1.0)
            results.append(result)
        return results

    @task(3)
    def save_conversation(self):
        """Test saving context - most common operation."""
        start_time = time.time()
        try:
            inputs = {"input": f"User message {random.randint(1, 1000)}"}
            outputs = {"output": f"AI response {random.randint(1, 1000)}"}

            self.memory.save_context(inputs, outputs)

            elapsed = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="RecallBricks",
                name="save_context",
                response_time=elapsed,
                response_length=len(inputs["input"]) + len(outputs["output"]),
                exception=None,
                context={}
            )
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="RecallBricks",
                name="save_context",
                response_time=elapsed,
                response_length=0,
                exception=e,
                context={}
            )

    @task(5)
    def load_memory(self):
        """Test loading memory - most common read operation."""
        start_time = time.time()
        try:
            inputs = {"input": f"Query {random.randint(1, 100)}"}

            result = self.memory.load_memory_variables(inputs)

            elapsed = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="RecallBricks",
                name="load_memory",
                response_time=elapsed,
                response_length=len(str(result)),
                exception=None,
                context={}
            )
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="RecallBricks",
                name="load_memory",
                response_time=elapsed,
                response_length=0,
                exception=e,
                context={}
            )

    @task(1)
    def full_conversation_cycle(self):
        """Test a complete conversation cycle."""
        start_time = time.time()
        try:
            # Load context
            inputs = {"input": "What do you know about me?"}
            self.memory.load_memory_variables(inputs)

            # Save response
            outputs = {"output": "Here's what I know..."}
            self.memory.save_context(inputs, outputs)

            elapsed = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="RecallBricks",
                name="full_cycle",
                response_time=elapsed,
                response_length=100,
                exception=None,
                context={}
            )
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="RecallBricks",
                name="full_cycle",
                response_time=elapsed,
                response_length=0,
                exception=e,
                context={}
            )


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Track system metrics during load test."""
    print(f"\n{'='*60}")
    print("LOAD TEST STARTING")
    print(f"{'='*60}")
    print(f"Start time: {datetime.now()}")
    print(f"Process ID: {os.getpid()}")
    print(f"Initial Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"{'='*60}\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Report system metrics after load test."""
    process = psutil.Process()
    print(f"\n{'='*60}")
    print("LOAD TEST COMPLETED")
    print(f"{'='*60}")
    print(f"End time: {datetime.now()}")
    print(f"Final Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"CPU Usage: {process.cpu_percent()}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Quick standalone test
    print("Running quick load test...")
    import subprocess
    subprocess.run([
        "locust",
        "-f", __file__,
        "--headless",
        "-u", "100",  # 100 users
        "-r", "10",   # Spawn 10 users per second
        "--run-time", "30s",
        "--host", "http://localhost"
    ])
