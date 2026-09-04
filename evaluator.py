import json
from typing import Dict, List
from agent import Agent
from eval_dataset import GOLDEN_DATASET
from logger import log_event

class AgentEvaluator:
    def __init__(self):
        # Use a fresh agent for each eval run to avoid state leakage
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
            
            # Check 1: Did it use the expected tool?
            # (We can check this by looking at the last few log events or agent memory)
            # For simplicity, we'll check if the response contains expected keywords
            # and infer tool usage from that.
            
            response_lower = response.lower()
            keywords_found = [kw.lower() for kw in expected_keywords if kw.lower() in response_lower]
            
            keyword_score = len(keywords_found) / len(expected_keywords)
            
            # Simple pass criteria: at least 50% of expected keywords are present
            passed = keyword_score >= 0.5
            
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
                "score": keyword_score,
                "passed": passed,
                "status": status
            }
            
            results.append(result)
            print(f"  {status} (Score: {keyword_score:.2f})")
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