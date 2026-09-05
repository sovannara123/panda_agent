import json
import re
from typing import Dict, List
from agent import Agent
from eval_dataset import GOLDEN_DATASET
from logger import log_event
from llm_adapters import create_llm_client, MockLLMClient
from config import get_config

class AgentEvaluator:
    def __init__(self, use_mock_llm: bool = True):
        # Use a fresh agent for each eval run to avoid state leakage
        if use_mock_llm:
            self.agent = Agent(llm_client=MockLLMClient(get_config().MODEL_NAME))
        else:
            self.agent = Agent()
        
    def run_evaluation(self) -> Dict:
        """Run all test cases and return a score report."""
        
        results = []
        total_tests = len(GOLDEN_DATASET)
        passed_tests = 0
        
        print(f"Starting evaluation of {total_tests} test cases...\n")
        
        for test_case in GOLDEN_DATASET:
            test_id = test_case["id"]
            user_input = test_case["input"]
            expected_tool = test_case["expected_tool"]
            expected_keywords = test_case["expected_keywords"]
            
            print(f"Running: {test_id} - {test_case['description']}")
            
            # Get agent response
            response = self.agent.respond(user_input)
            
            # Evaluate with flexible matching
            score, passed, keywords_found = self._evaluate_response(
                response, expected_keywords, expected_tool
            )
            
            if passed:
                passed_tests += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
                
            result = {
                "test_id": test_id,
                "input": user_input,
                "response": response,
                "expected_keywords": expected_keywords,
                "keywords_found": keywords_found,
                "score": score,
                "passed": passed,
                "status": status
            }
            
            results.append(result)
            print(f"  {status} (Score: {score:.2f})")
            print(f"  Response: {response[:100]}...\n")
            
        # Calculate final metrics
        overall_score = (passed_tests / total_tests) * 100
        
        report = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "overall_score": overall_score,
            "detailed_results": results
        }
        
        log_event("evaluation_complete", {
            "score": overall_score,
            "passed": passed_tests,
            "total": total_tests
        })
        
        return report

    def _evaluate_response(self, response: str, expected_keywords: List[str], expected_tool: str) -> tuple:
        """Evaluate response with flexible matching."""
        response_lower = response.lower()
        
        # Check for keyword matches (flexible: substrings, word boundaries)
        keywords_found = []
        for kw in expected_keywords:
            kw_lower = kw.lower()
            # Check exact substring
            if kw_lower in response_lower:
                keywords_found.append(kw)
            # Check word boundaries for multi-word keywords
            elif re.search(rf'\b{re.escape(kw_lower)}\b', response_lower):
                keywords_found.append(kw)
        
        keyword_score = len(keywords_found) / len(expected_keywords) if expected_keywords else 1.0
        
        # Additional checks for meaningful response
        has_content = len(response.strip()) > 10
        not_fallback = "unavailable" not in response_lower or "error" not in response_lower
        
        # Pass if: good keyword match OR (meaningful content AND no fallback indicators)
        passed = keyword_score >= 0.5 or (has_content and not_fallback and keyword_score > 0)
        
        return keyword_score, passed, keywords_found

    def save_report(self, report: Dict, filename: str = "eval_report.json"):
        """Save the evaluation report to a file."""
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Evaluation report saved to {filename}")


if __name__ == "__main__":
    evaluator = AgentEvaluator()
    report = evaluator.run_evaluation()
    evaluator.save_report(report)
    
    print("\n" + "="*50)
    print(f"FINAL SCORE: {report['overall_score']:.1f}% ({report['passed_tests']}/{report['total_tests']})")
    print("="*50)