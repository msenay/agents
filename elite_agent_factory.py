#!/usr/bin/env python3
"""
🚀 Elite Agent Factory
======================

Elite Coder Agent kullanarak süper güçlü agent factory sistemi.
Her task'ı hatırlar, öğrenir ve sürekli gelişir.

Features:
- 🧠 Elite Coder Agent with advanced memory
- 📚 Cross-task learning and pattern recognition
- 🔄 Continuous improvement
- ⚡ Multi-task batch processing
- 🎯 Task-aware agent creation
- 🏆 Quality assessment and feedback learning

Usage:
    factory = EliteAgentFactory(api_key="your-key")
    result = factory.create_agent("Build a sentiment analysis agent")
    
Multi-task:
    results = factory.create_multiple_agents([
        "Create data processor",
        "Build web scraper", 
        "Make ML classifier"
    ])
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from elite_coder_agent import EliteCoderAgent, AgentCreationResult, TaskContext
from agent_factory import AgentRequest, AgentResult


@dataclass
class EliteAgentRequest:
    """Enhanced agent request with learning context"""
    task: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    temperature: float = 0.1
    tools: Optional[List[str]] = None
    requirements: Optional[List[str]] = None
    complexity: str = "intermediate"  # basic, intermediate, advanced, expert
    learning_enabled: bool = True
    session_id: Optional[str] = None
    

@dataclass 
class EliteAgentResult:
    """Enhanced agent result with learning metrics"""
    success: bool
    agent_code: str
    test_code: str
    review_feedback: str
    test_results: str
    file_path: Optional[str] = None
    errors: Optional[List[str]] = None
    
    # Elite features
    task_id: str = ""
    quality_score: float = 0.0
    complexity_score: float = 0.0
    patterns_used: List[str] = None
    creation_time: float = 0.0
    learning_applied: bool = False
    similar_tasks_found: int = 0


class EliteAgentFactory:
    """Elite Agent Factory with advanced learning and memory"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o", session_id: Optional[str] = None):
        
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
        # Initialize Elite Coder Agent
        self.elite_coder = EliteCoderAgent(
            api_key=api_key,
            base_url=base_url,
            model=model,
            session_id=session_id
        )
        
        # Factory metrics
        self.factory_stats = {
            "total_agents_created": 0,
            "successful_creations": 0,
            "failed_creations": 0,
            "average_quality_score": 0.0,
            "total_creation_time": 0.0,
            "patterns_discovered": 0,
            "learning_iterations": 0
        }
        
        logging.info(f"🚀 Elite Agent Factory initialized with session: {self.elite_coder.session_id}")
    
    def create_agent(self, request: EliteAgentRequest) -> EliteAgentResult:
        """Create a single agent with elite capabilities"""
        
        start_time = time.time()
        
        print(f"🚀 ELITE AGENT FACTORY: Creating agent for task")
        print(f"Task: {request.task}")
        print(f"Complexity: {request.complexity}")
        print(f"Tools: {request.tools}")
        print(f"Requirements: {request.requirements}")
        print("=" * 70)
        
        try:
            # Step 1: Create agent using Elite Coder
            print("🧠 Step 1: Elite Coder analyzing and creating agent...")
            coder_result = self.elite_coder.create_agent(
                task_description=request.task,
                requirements=request.requirements,
                tools=request.tools,
                complexity=request.complexity
            )
            
            if not coder_result.success:
                return self._create_failed_result(coder_result, start_time)
            
            # Step 2: Generate comprehensive tests
            print("🧪 Step 2: Generating comprehensive test suite...")
            test_code = self._generate_tests(coder_result.agent_code, request.task)
            
            # Step 3: Perform code review with learning
            print("🔍 Step 3: Conducting intelligent code review...")
            review_feedback = self._review_with_memory(coder_result.agent_code, request.task, coder_result)
            
            # Step 4: Test agent functionality
            print("⚡ Step 4: Testing agent functionality...")
            test_results = self._test_agent_functionality(coder_result.agent_code, request.task)
            
            # Step 5: Save agent with metadata
            print("💾 Step 5: Saving agent with learning metadata...")
            file_path = self._save_elite_agent(coder_result, request, test_code, review_feedback)
            
            # Step 6: Apply learning and feedback
            print("📚 Step 6: Applying learning and updating patterns...")
            learning_applied = self._apply_learning_feedback(coder_result, request)
            
            # Step 7: Update factory statistics
            self._update_factory_stats(coder_result, start_time)
            
            creation_time = time.time() - start_time
            print(f"✅ Elite agent created successfully in {creation_time:.2f}s!")
            print(f"   Quality Score: {coder_result.quality_score:.2f}")
            print(f"   Patterns Used: {len(coder_result.patterns_used)}")
            print(f"   Code Length: {coder_result.code_length} characters")
            
            return EliteAgentResult(
                success=True,
                agent_code=coder_result.agent_code,
                test_code=test_code,
                review_feedback=review_feedback,
                test_results=test_results,
                file_path=file_path,
                task_id=coder_result.task_id,
                quality_score=coder_result.quality_score,
                complexity_score=coder_result.complexity_score,
                patterns_used=coder_result.patterns_used,
                creation_time=creation_time,
                learning_applied=learning_applied,
                similar_tasks_found=len(self.elite_coder._find_similar_tasks(
                    self.elite_coder._analyze_task(request.task, request.requirements, request.tools, request.complexity)
                ))
            )
            
        except Exception as e:
            logging.error(f"❌ Elite agent creation failed: {str(e)}")
            self.factory_stats["failed_creations"] += 1
            
            return EliteAgentResult(
                success=False,
                agent_code="",
                test_code="",
                review_feedback="",
                test_results=f"Creation failed: {str(e)}",
                errors=[str(e)],
                creation_time=time.time() - start_time
            )
    
    def create_multiple_agents(self, requests: List[EliteAgentRequest]) -> List[EliteAgentResult]:
        """Create multiple agents with intelligent batch learning"""
        
        print(f"🎯 ELITE BATCH CREATION: Creating {len(requests)} agents")
        print("=" * 70)
        
        # Convert to task descriptions for Elite Coder batch processing
        task_descriptions = [req.task for req in requests]
        batch_requirements = [req.requirements for req in requests]
        
        # Use Elite Coder's intelligent batch processing
        coder_results = self.elite_coder.create_multiple_agents(
            task_descriptions, batch_requirements
        )
        
        # Convert to Elite Factory results
        elite_results = []
        
        for i, (request, coder_result) in enumerate(zip(requests, coder_results)):
            print(f"\n🔄 Processing agent {i+1}/{len(requests)}: {request.task[:50]}...")
            
            if coder_result.success:
                # Generate additional components for successful agents
                test_code = self._generate_tests(coder_result.agent_code, request.task)
                review_feedback = self._review_with_memory(coder_result.agent_code, request.task, coder_result)
                test_results = self._test_agent_functionality(coder_result.agent_code, request.task)
                file_path = self._save_elite_agent(coder_result, request, test_code, review_feedback)
                learning_applied = self._apply_learning_feedback(coder_result, request)
                
                elite_result = EliteAgentResult(
                    success=True,
                    agent_code=coder_result.agent_code,
                    test_code=test_code,
                    review_feedback=review_feedback,
                    test_results=test_results,
                    file_path=file_path,
                    task_id=coder_result.task_id,
                    quality_score=coder_result.quality_score,
                    complexity_score=coder_result.complexity_score,
                    patterns_used=coder_result.patterns_used,
                    creation_time=coder_result.creation_time,
                    learning_applied=learning_applied
                )
            else:
                elite_result = EliteAgentResult(
                    success=False,
                    agent_code="",
                    test_code="",
                    review_feedback="",
                    test_results=f"Creation failed: {coder_result.errors}",
                    errors=coder_result.errors,
                    creation_time=coder_result.creation_time
                )
            
            elite_results.append(elite_result)
            self._update_factory_stats(coder_result, 0)
        
        successful_count = sum(1 for r in elite_results if r.success)
        print(f"\n✅ Batch creation completed: {successful_count}/{len(requests)} successful")
        
        # Apply batch learning
        self._apply_batch_learning(elite_results)
        
        return elite_results
    
    def _generate_tests(self, agent_code: str, task_description: str) -> str:
        """Generate comprehensive tests using Elite Coder's memory"""
        
        # Use Elite Coder to generate tests with memory context
        test_prompt = f"""
        Create comprehensive unit tests for this agent code:
        
        Task: {task_description}
        
        Agent Code:
        ```python
        {agent_code}
        ```
        
        Generate complete test code with:
        1. pytest framework
        2. Mock external dependencies
        3. Test edge cases and error scenarios
        4. Test main functionality
        5. Performance tests if applicable
        
        Use patterns from similar previous tasks for optimal testing.
        """
        
        try:
            response = self.elite_coder.agent.invoke(test_prompt)
            test_code = response['messages'][-1].content
            
            # Extract Python code
            if "```python" in test_code:
                test_code = test_code.split("```python")[1].split("```")[0].strip()
            elif "```" in test_code:
                test_code = test_code.split("```")[1].split("```")[0].strip()
            
            return test_code
            
        except Exception as e:
            logging.warning(f"Test generation failed: {e}")
            return f"# Test generation failed: {e}\n# TODO: Add manual tests"
    
    def _review_with_memory(self, agent_code: str, task_description: str, 
                          coder_result: AgentCreationResult) -> str:
        """Conduct code review with memory of previous patterns"""
        
        review_prompt = f"""
        Review this agent code with context from previous similar tasks:
        
        Task: {task_description}
        Quality Score: {coder_result.quality_score:.2f}
        Patterns Used: {coder_result.patterns_used}
        Complexity: {coder_result.complexity_score:.2f}
        
        Code:
        ```python
        {agent_code}
        ```
        
        Provide detailed review covering:
        1. Code quality vs previous similar agents
        2. Security considerations
        3. Performance optimization opportunities
        4. Best practices compliance
        5. Maintainability assessment
        6. Comparison with learned patterns
        7. Recommendations for improvement
        
        Use memory of previous reviews for consistency.
        """
        
        try:
            response = self.elite_coder.agent.invoke(review_prompt)
            return response['messages'][-1].content
            
        except Exception as e:
            logging.warning(f"Code review failed: {e}")
            return f"Code review failed: {e}"
    
    def _test_agent_functionality(self, agent_code: str, task_description: str) -> str:
        """Test agent functionality with intelligent analysis"""
        
        test_prompt = f"""
        Analyze this agent code for functionality and correctness:
        
        Task: {task_description}
        
        Code:
        ```python
        {agent_code}
        ```
        
        Provide analysis:
        1. Will this code execute without errors?
        2. Does it fulfill the task requirements?
        3. Are there any runtime issues?
        4. How well does it handle edge cases?
        5. Overall functionality assessment (1-10)
        
        Based on memory of similar agent testing.
        """
        
        try:
            response = self.elite_coder.agent.invoke(test_prompt)
            return response['messages'][-1].content
            
        except Exception as e:
            logging.warning(f"Functionality testing failed: {e}")
            return f"Functionality testing failed: {e}"
    
    def _save_elite_agent(self, coder_result: AgentCreationResult, request: EliteAgentRequest,
                         test_code: str, review_feedback: str) -> str:
        """Save agent with elite metadata"""
        
        import tempfile
        from pathlib import Path
        
        # Create filename from task
        safe_task = "".join(c for c in request.task if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_task = safe_task.replace(' ', '_').lower()[:50]
        
        # Create temporary file (in real implementation, use proper directory)
        temp_dir = tempfile.gettempdir()
        agent_file = Path(temp_dir) / f"elite_agent_{safe_task}_{coder_result.task_id}.py"
        
        # Enhanced agent code with metadata
        enhanced_code = f'''"""
Elite Agent: {request.task}
===============================

Generated by Elite Agent Factory
Task ID: {coder_result.task_id}
Quality Score: {coder_result.quality_score:.2f}
Complexity Score: {coder_result.complexity_score:.2f}
Patterns Used: {', '.join(coder_result.patterns_used)}
Creation Time: {coder_result.creation_time:.2f}s
Session ID: {self.elite_coder.session_id}

Requirements: {', '.join(request.requirements or [])}
Tools: {', '.join(request.tools or [])}
"""

{coder_result.agent_code}
'''
        
        # Save to file
        try:
            with open(agent_file, 'w', encoding='utf-8') as f:
                f.write(enhanced_code)
            
            # Also save test file
            test_file = agent_file.parent / f"test_{agent_file.name}"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(f'"""\nUnit tests for: {request.task}\n"""\n\n{test_code}')
            
            return str(agent_file)
            
        except Exception as e:
            logging.warning(f"Could not save agent file: {e}")
            return ""
    
    def _apply_learning_feedback(self, coder_result: AgentCreationResult, 
                               request: EliteAgentRequest) -> bool:
        """Apply learning and feedback to improve future creations"""
        
        try:
            # Provide feedback to Elite Coder
            if coder_result.quality_score >= 0.8:
                feedback = f"High quality agent created for {request.task}. Patterns worked well."
                self.elite_coder.learn_from_feedback(
                    coder_result.task_id, feedback, coder_result.quality_score
                )
            elif coder_result.quality_score >= 0.6:
                feedback = f"Moderate quality agent for {request.task}. Some improvements needed."
                self.elite_coder.learn_from_feedback(
                    coder_result.task_id, feedback, coder_result.quality_score
                )
            else:
                feedback = f"Low quality agent for {request.task}. Significant improvements needed."
                self.elite_coder.learn_from_feedback(
                    coder_result.task_id, feedback, coder_result.quality_score
                )
            
            self.factory_stats["learning_iterations"] += 1
            return True
            
        except Exception as e:
            logging.warning(f"Learning feedback failed: {e}")
            return False
    
    def _apply_batch_learning(self, results: List[EliteAgentResult]):
        """Apply learning from batch results"""
        
        successful_results = [r for r in results if r.success and r.quality_score >= 0.7]
        
        if successful_results:
            avg_quality = sum(r.quality_score for r in successful_results) / len(successful_results)
            
            # Update factory patterns based on successful batch
            self.factory_stats["patterns_discovered"] += len(successful_results)
            
            logging.info(f"📚 Batch learning applied: {len(successful_results)} successful agents, avg quality: {avg_quality:.2f}")
    
    def _update_factory_stats(self, coder_result: AgentCreationResult, start_time: float):
        """Update factory statistics"""
        
        self.factory_stats["total_agents_created"] += 1
        
        if coder_result.success:
            self.factory_stats["successful_creations"] += 1
            
            # Update average quality score
            current_avg = self.factory_stats["average_quality_score"]
            total_successful = self.factory_stats["successful_creations"]
            new_avg = ((current_avg * (total_successful - 1)) + coder_result.quality_score) / total_successful
            self.factory_stats["average_quality_score"] = new_avg
        else:
            self.factory_stats["failed_creations"] += 1
        
        self.factory_stats["total_creation_time"] += coder_result.creation_time
    
    def _create_failed_result(self, coder_result: AgentCreationResult, start_time: float) -> EliteAgentResult:
        """Create a failed result from coder result"""
        
        return EliteAgentResult(
            success=False,
            agent_code="",
            test_code="",
            review_feedback="",
            test_results=f"Elite Coder failed: {coder_result.errors}",
            errors=coder_result.errors,
            task_id=coder_result.task_id,
            creation_time=time.time() - start_time
        )
    
    def get_factory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive factory statistics"""
        
        elite_stats = self.elite_coder.get_agent_statistics()
        
        factory_success_rate = (
            self.factory_stats["successful_creations"] / 
            self.factory_stats["total_agents_created"]
        ) if self.factory_stats["total_agents_created"] > 0 else 0
        
        avg_creation_time = (
            self.factory_stats["total_creation_time"] / 
            self.factory_stats["total_agents_created"]
        ) if self.factory_stats["total_agents_created"] > 0 else 0
        
        return {
            "factory_metrics": {
                "total_agents_created": self.factory_stats["total_agents_created"],
                "success_rate": factory_success_rate,
                "average_quality_score": self.factory_stats["average_quality_score"],
                "average_creation_time": avg_creation_time,
                "patterns_discovered": self.factory_stats["patterns_discovered"],
                "learning_iterations": self.factory_stats["learning_iterations"]
            },
            "elite_coder_metrics": elite_stats,
            "session_id": self.elite_coder.session_id,
            "model_used": self.model
        }
    
    def create_agent_simple(self, task: str, api_key: str, **kwargs) -> EliteAgentResult:
        """Simple interface for quick agent creation"""
        
        request = EliteAgentRequest(
            task=task,
            api_key=api_key,
            **kwargs
        )
        
        return self.create_agent(request)


def main():
    """Demo of Elite Agent Factory"""
    
    print("🚀 ELITE AGENT FACTORY DEMO")
    print("=" * 60)
    
    # Initialize Elite Factory
    factory = EliteAgentFactory(
        api_key="your-openai-api-key",  # Replace with real key
        model="gpt-4o-mini"
    )
    
    # Single elite agent creation
    print("\n🎯 Creating elite sentiment analysis agent...")
    
    request = EliteAgentRequest(
        task="Create a sentiment analysis agent that processes text and returns detailed sentiment analysis",
        api_key="your-openai-api-key",
        requirements=["Handle multiple languages", "Return confidence scores", "Batch processing"],
        tools=["transformers", "torch", "numpy"],
        complexity="advanced"
    )
    
    result = factory.create_agent(request)
    
    if result.success:
        print(f"✅ Elite agent created successfully!")
        print(f"   Quality Score: {result.quality_score:.2f}")
        print(f"   Complexity Score: {result.complexity_score:.2f}")
        print(f"   Patterns Used: {len(result.patterns_used or [])}")
        print(f"   Similar Tasks Found: {result.similar_tasks_found}")
        print(f"   Learning Applied: {result.learning_applied}")
        print(f"   File: {result.file_path}")
    else:
        print(f"❌ Failed: {result.errors}")
    
    # Multiple elite agents
    print(f"\n🎨 Creating multiple elite agents...")
    
    requests = [
        EliteAgentRequest(
            task="Create a web scraper for e-commerce data with rate limiting",
            api_key="your-openai-api-key",
            tools=["requests", "beautifulsoup4", "selenium"],
            complexity="advanced"
        ),
        EliteAgentRequest(
            task="Build a CSV data processor with statistical analysis",
            api_key="your-openai-api-key",
            tools=["pandas", "numpy", "matplotlib"],
            complexity="intermediate"
        ),
        EliteAgentRequest(
            task="Create an email classification agent with ML",
            api_key="your-openai-api-key",
            tools=["scikit-learn", "nltk", "email"],
            complexity="advanced"
        )
    ]
    
    results = factory.create_multiple_agents(requests)
    
    successful = sum(1 for r in results if r.success)
    print(f"Batch results: {successful}/{len(results)} successful")
    
    # Show elite statistics
    stats = factory.get_factory_statistics()
    print(f"\n📊 ELITE FACTORY STATISTICS:")
    print(f"Factory Success Rate: {stats['factory_metrics']['success_rate']:.1%}")
    print(f"Average Quality Score: {stats['factory_metrics']['average_quality_score']:.2f}")
    print(f"Average Creation Time: {stats['factory_metrics']['average_creation_time']:.2f}s")
    print(f"Patterns Discovered: {stats['factory_metrics']['patterns_discovered']}")
    print(f"Learning Iterations: {stats['factory_metrics']['learning_iterations']}")
    print(f"Elite Coder Tasks: {stats['elite_coder_metrics']['total_tasks_completed']}")
    print(f"Elite Coder Success Rate: {stats['elite_coder_metrics']['success_rate']:.1%}")


if __name__ == "__main__":
    main()