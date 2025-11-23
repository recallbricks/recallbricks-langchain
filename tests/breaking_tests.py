"""
BREAKING TESTS - Attempts to break RecallBricksMemory
These tests demonstrate security vulnerabilities and edge cases.

Run with: python tests/breaking_tests.py

WARNING: These tests are designed to EXPOSE vulnerabilities!
"""

import unittest
import threading
import time
import os
import uuid
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
from recallbricks_langchain import RecallBricksMemory


class BreakingSecurityTests(unittest.TestCase):
    """Tests designed to break security."""

    def setUp(self):
        os.environ['RECALLBRICKS_SERVICE_TOKEN'] = 'test-token-12345'
        self.valid_user_id = str(uuid.uuid4())

    def test_01_http_downgrade_attack(self):
        """VULNERABILITY: Accepts HTTP URLs (man-in-the-middle risk)"""
        print("\n[TEST 1] HTTP Downgrade Attack")
        print("="*60)

        try:
            # This SHOULD fail but doesn't!
            memory = RecallBricksMemory(
                agent_id="test",
                api_url="http://malicious-attacker.com",  # HTTP not HTTPS!
                user_id=self.valid_user_id
            )
            print("[FAIL] HTTP URL accepted (security risk!)")
            print(f"   API URL: {memory.api_url}")
            return False
        except ValueError as e:
            print("[OK] HTTP URL rejected")
            return True

    def test_02_sql_injection_user_id(self):
        """VULNERABILITY: No validation on user_id format"""
        print("\n[TEST 2] SQL Injection in user_id")
        print("="*60)

        malicious_payloads = [
            "'; DROP TABLE users--",
            "1' OR '1'='1",
            "admin'--",
            "../../../etc/passwd",
            "<script>alert('xss')</script>",
        ]

        vulnerable_count = 0

        for payload in malicious_payloads:
            try:
                memory = RecallBricksMemory(
                    agent_id="test",
                    user_id=payload  # Malicious user_id
                )
                print(f"[FAIL]: Accepted malicious user_id: {payload[:30]}")
                vulnerable_count += 1
            except ValueError:
                print(f"[OK]: Rejected malicious user_id: {payload[:30]}")

        if vulnerable_count > 0:
            print(f"\n[VULNERABLE]: {vulnerable_count}/{len(malicious_payloads)} payloads accepted!")
            return False
        return True

    def test_03_token_leak_in_exception(self):
        """VULNERABILITY: Service token may leak in exceptions"""
        print("\n[TEST 3] Token Leak in Exception Traces")
        print("="*60)

        # Capture stderr to check for token leaks
        captured_output = StringIO()

        with patch('requests.post') as mock_post:
            # Force an exception that might leak the token
            mock_post.side_effect = Exception("API Error with headers: ...")

            memory = RecallBricksMemory(
                agent_id="test",
                user_id=self.valid_user_id,
                service_token="SECRET-TOKEN-12345"
            )

            try:
                memory.save_context({"input": "test"}, {"output": "test"})
            except Exception as e:
                exception_str = str(e)
                if "SECRET" in exception_str or "12345" in exception_str:
                    print("[FAIL]: Token leaked in exception!")
                    print(f"   Exception: {exception_str[:100]}")
                    return False

        print("[OK]: No token leak detected (in this test)")
        return True

    def test_04_race_condition_circuit_breaker(self):
        """VULNERABILITY: Race condition in circuit breaker state check"""
        print("\n[TEST 4] Circuit Breaker Race Condition")
        print("="*60)

        with patch('requests.get') as mock_get:
            # Setup to fail initially
            mock_get.side_effect = Exception("Connection failed")

            memory = RecallBricksMemory(
                agent_id="test",
                user_id=self.valid_user_id,
                circuit_breaker_threshold=2
            )

            # Force circuit to open
            for _ in range(3):
                try:
                    memory.load_memory_variables({"input": "test"})
                except:
                    pass

            # Now race condition: check state outside lock
            results = []
            errors = []

            def worker(thread_id):
                try:
                    # This should fail if circuit is open
                    # But race condition might let some through
                    memory.load_memory_variables({"input": "test"})
                    results.append(f"Thread {thread_id} succeeded (race!)")
                except Exception as e:
                    errors.append(f"Thread {thread_id} blocked correctly")

            # Launch 50 threads simultaneously
            threads = []
            for i in range(50):
                t = threading.Thread(target=worker, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            if results:
                print(f"[FAIL]: {len(results)} requests bypassed circuit breaker!")
                print(f"   This indicates a race condition vulnerability")
                return False
            else:
                print(f"[OK]: All {len(errors)} requests blocked by circuit")
                return True

    def test_05_memory_exhaustion_attack(self):
        """ATTACK: Try to exhaust memory with huge payload"""
        print("\n[TEST 5] Memory Exhaustion Attack")
        print("="*60)

        with patch('requests.post') as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {})

            memory = RecallBricksMemory(
                agent_id="test",
                user_id=self.valid_user_id,
                max_text_length=100000  # 100KB limit
            )

            # Try to send 10MB (should be truncated to 100KB)
            huge_text = "A" * (10 * 1024 * 1024)  # 10MB

            try:
                memory.save_context(
                    {"input": huge_text},
                    {"output": huge_text}
                )

                # Check if it was actually truncated
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                actual_size = len(payload['text'])

                if actual_size > 200000:  # Should be max 200KB (100KB * 2)
                    print(f"[FAIL]: Payload too large ({actual_size} bytes)")
                    return False
                else:
                    print(f"[OK]: Payload truncated to {actual_size} bytes")
                    return True
            except Exception as e:
                print(f"[OK]: Exception raised for huge payload: {e}")
                return True

    def test_06_rate_limit_attack(self):
        """ATTACK: Spam API with rapid requests (no rate limiting)"""
        print("\n[TEST 6] Rate Limit Attack")
        print("="*60)

        with patch('requests.get') as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"memories": []}
            )

            memory = RecallBricksMemory(
                agent_id="test",
                user_id=self.valid_user_id
            )

            # Try to make 1000 requests rapidly
            start = time.time()
            count = 0

            for i in range(1000):
                try:
                    memory.load_memory_variables({"input": f"query {i}"})
                    count += 1
                except Exception:
                    break

            elapsed = time.time() - start
            rate = count / elapsed if elapsed > 0 else 0

            print(f"Completed {count} requests in {elapsed:.2f}s")
            print(f"Rate: {rate:.0f} requests/second")

            if count == 1000:
                print("[FAIL]: No rate limiting! All 1000 requests succeeded")
                print("   This could lead to API abuse and cost attacks")
                return False
            else:
                print(f"[OK]: Rate limiting kicked in after {count} requests")
                return True

    def test_07_empty_none_payloads(self):
        """EDGE CASE: Empty and None values"""
        print("\n[TEST 7] Empty/None Payload Handling")
        print("="*60)

        test_cases = [
            ({"input": ""}, {"output": ""}, "empty strings"),
            ({"input": None}, {"output": None}, "None values"),
            ({"input": " "}, {"output": " "}, "whitespace only"),
            ({}, {}, "empty dicts"),
        ]

        issues = 0

        for inputs, outputs, description in test_cases:
            with patch('requests.post') as mock_post:
                mock_post.return_value = MagicMock(status_code=200, json=lambda: {})

                memory = RecallBricksMemory(
                    agent_id="test",
                    user_id=self.valid_user_id
                )

                try:
                    memory.save_context(inputs, outputs)

                    # Check what was actually sent
                    if mock_post.called:
                        payload = mock_post.call_args[1]['json']
                        text = payload.get('text', '')

                        if text in ["Human: \nAI: ", "Human: None\nAI: None", ""]:
                            print(f"[WARN]: {description} created useless memory: '{text}'")
                            issues += 1
                        else:
                            print(f"[OK]: {description} handled: '{text}'")
                except Exception as e:
                    print(f"[OK]: {description} rejected: {e}")

        if issues > 0:
            print(f"\n[WARN] {issues} edge cases created useless data")
            return False
        return True

    def test_08_unicode_bombs(self):
        """ATTACK: Unicode edge cases and bombs"""
        print("\n[TEST 8] Unicode Attack Vectors")
        print("="*60)

        # Various Unicode attack vectors
        unicode_bombs = [
            ("Zalgo text", "H̷̢̖̓͌e̸̢̛͆l̴͎̈́l̴̰̈́ǒ̶̧"),
            ("Right-to-left override", "\u202E" + "sneaky"),
            ("Zero-width chars", "hel\u200Blo"),
            ("Emoji spam", "🔥" * 10000),
            ("Combining chars", "e" + "\u0301" * 100),
            ("Homoglyphs", "раypal.com"),  # Cyrillic 'a'
        ]

        issues = 0

        for description, payload in unicode_bombs:
            with patch('requests.post') as mock_post:
                mock_post.return_value = MagicMock(status_code=200, json=lambda: {})

                memory = RecallBricksMemory(
                    agent_id="test",
                    user_id=self.valid_user_id
                )

                try:
                    memory.save_context({"input": payload}, {"output": "ok"})

                    if mock_post.called:
                        sent_payload = mock_post.call_args[1]['json']['text']
                        print(f"[OK] {description}: Accepted ({len(sent_payload)} chars)")
                except Exception as e:
                    print(f"[FAIL] {description}: Crashed! {e}")
                    issues += 1

        return issues == 0

    def test_09_timezone_attack(self):
        """VULNERABILITY: Timezone issues in datetime.now()"""
        print("\n[TEST 9] Timezone Attack on Circuit Breaker")
        print("="*60)

        # This test demonstrates timezone issues
        # In production, datetime.now() could be different across servers

        memory = RecallBricksMemory(
            agent_id="test",
            user_id=self.valid_user_id
        )

        # Check if datetime is timezone-aware
        cb = memory.circuit_breaker

        # Trigger a failure to set last_failure_time
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Fail")

            try:
                memory.load_memory_variables({"input": "test"})
            except:
                pass

        if cb.last_failure_time and cb.last_failure_time.tzinfo is None:
            print("[FAIL]: Using timezone-naive datetime")
            print("   This will cause issues in multi-region deployments")
            return False
        else:
            print("[OK]: Using timezone-aware datetime (or not set)")
            return True

    def test_10_connection_exhaustion(self):
        """ATTACK: Open many instances to exhaust connections"""
        print("\n[TEST 10] Connection Exhaustion Attack")
        print("="*60)

        # Try to create 1000 memory instances
        instances = []

        try:
            for i in range(1000):
                memory = RecallBricksMemory(
                    agent_id=f"agent-{i}",
                    user_id=str(uuid.uuid4())
                )
                instances.append(memory)

            print(f"[FAIL]: Created {len(instances)} instances")
            print("   No connection pooling - each instance creates new connections")
            print("   This could exhaust system resources")
            return False

        except Exception as e:
            print(f"[OK]: Connection limit reached at {len(instances)} instances")
            return True
        finally:
            instances.clear()


def run_breaking_tests():
    """Run all breaking tests and generate security report."""
    print("\n" + "="*60)
    print("RECALLBRICKS SECURITY BREAKING TESTS")
    print("="*60)
    print("Attempting to break the system to find vulnerabilities...")
    print("="*60 + "\n")

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(BreakingSecurityTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Security report
    print("\n" + "="*60)
    print("SECURITY AUDIT RESULTS")
    print("="*60)

    total_tests = result.testsRun
    passed = total_tests - len(result.failures) - len(result.errors)
    failed = len(result.failures) + len(result.errors)

    print(f"Tests run: {total_tests}")
    print(f"Vulnerabilities found: {failed}")
    print(f"Security controls working: {passed}")

    security_score = (passed / total_tests * 100) if total_tests > 0 else 0

    print(f"\nSECURITY SCORE: {security_score:.0f}/100")

    if security_score >= 90:
        print("[OK] EXCELLENT - Enterprise ready")
    elif security_score >= 70:
        print("[WARN] GOOD - Minor issues to fix")
    elif security_score >= 50:
        print("[WARN] FAIR - Several security gaps")
    else:
        print("[FAIL] POOR - NOT production ready")

    print("="*60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_breaking_tests()
    sys.exit(0 if success else 1)
