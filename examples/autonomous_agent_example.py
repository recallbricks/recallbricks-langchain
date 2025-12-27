"""
Autonomous Agent Example: LangChain with Working Memory and Goal Tracking (v1.3.0)

This example demonstrates the new autonomous agent features:
- Working memory sessions for multi-step reasoning
- Goal tracking across multiple steps
- Metacognition assessment (quality and uncertainty)
- Context manager and decorator patterns

These features enable LangChain agents to:
- Maintain task-specific context during complex operations
- Track progress through multi-step goals
- Self-assess response quality and uncertainty
"""

from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
from recallbricks_langchain import RecallBricksMemory
import os
import uuid


def create_autonomous_memory() -> RecallBricksMemory:
    """
    Create a RecallBricksMemory instance with autonomous features enabled.

    Returns:
        RecallBricksMemory configured for autonomous agent operation
    """
    return RecallBricksMemory(
        agent_id="autonomous-agent",
        api_key=os.getenv("RECALLBRICKS_API_KEY"),
        enable_autonomous=True,  # Enable autonomous features
        autonomous_features={
            "working_memory_ttl": 3600,  # 1 hour TTL for working memory
            "goal_tracking_enabled": True,
            "metacognition_enabled": True,
            "confidence_threshold": 0.7  # Quality threshold
        },
        organized=True,
        enable_metrics=True
    )


def demonstrate_working_memory():
    """
    Demonstrate working memory sessions for multi-step reasoning.

    Working memory provides short-term, task-specific memory that
    persists only for the duration of a task or session.
    """
    print("\n" + "=" * 60)
    print("WORKING MEMORY DEMONSTRATION")
    print("=" * 60)

    memory = create_autonomous_memory()

    # Use context manager for automatic cleanup
    print("\nUsing context manager for working memory:")
    with memory.with_working_memory("research-task-001") as session:
        print(f"  Created session: {session['session_id']}")

        # Simulate multi-step research process
        memory.add_to_working_memory(
            "research-task-001",
            "Step 1: User asked about Python async performance"
        )
        memory.add_to_working_memory(
            "research-task-001",
            "Step 2: Found that asyncio is suitable for I/O-bound tasks"
        )
        memory.add_to_working_memory(
            "research-task-001",
            "Step 3: Identified multiprocessing for CPU-bound tasks"
        )

        # Retrieve working memory contents
        items = memory.get_working_memory("research-task-001")
        print(f"  Working memory contains {len(items)} items")
        for item in items:
            print(f"    - {item['content']}")

    print("  Session automatically cleaned up!")

    # Manual session management
    print("\nManual session management:")
    session = memory.create_working_memory_session("analysis-task-002")
    print(f"  Created session: {session['session_id']}")

    memory.add_to_working_memory(
        "analysis-task-002",
        "Analyzing user preferences from conversation history"
    )

    # End and optionally persist to long-term memory
    memory.end_working_memory_session("analysis-task-002", persist=True)
    print("  Session ended and persisted to long-term memory")


def demonstrate_goal_tracking():
    """
    Demonstrate goal tracking for multi-step task management.

    Goal tracking enables agents to maintain awareness of their
    objectives and track progress through complex tasks.
    """
    print("\n" + "=" * 60)
    print("GOAL TRACKING DEMONSTRATION")
    print("=" * 60)

    memory = create_autonomous_memory()

    # Define a multi-step goal
    goal_steps = [
        "Gather user requirements",
        "Search knowledge base for relevant information",
        "Synthesize findings into coherent response",
        "Validate response quality",
        "Deliver final answer"
    ]

    print("\nTracking multi-step goal:")
    goal = memory.track_goal("answer-complex-question", goal_steps)
    print(f"  Goal: {goal['goal_id']}")
    print(f"  Steps: {goal['total_steps']}")
    print(f"  Progress: {goal['progress']*100:.0f}%")

    # Simulate completing steps
    print("\nCompleting steps:")
    for i in range(len(goal_steps)):
        status = memory.complete_goal_step("answer-complex-question")
        print(f"  Step {i+1}: '{goal_steps[i]}' - {status['progress']*100:.0f}% complete")

    final_status = memory.get_goal_status("answer-complex-question")
    print(f"\nFinal status: {final_status['status']}")

    # Demonstrate decorator pattern
    print("\nUsing decorator for goal tracking:")

    @memory.with_goal_tracking("data-analysis", ["fetch", "process", "report"])
    def analyze_data():
        """Function with automatic goal tracking."""
        print("  Executing analysis function...")
        return {"result": "Analysis complete", "metrics": {"accuracy": 0.95}}

    result = analyze_data()
    print(f"  Result: {result}")

    analysis_status = memory.get_goal_status("data-analysis")
    print(f"  Goal completed: {analysis_status['status'] == 'completed'}")


def demonstrate_metacognition():
    """
    Demonstrate metacognition features: quality assessment and uncertainty quantification.

    Metacognition enables agents to evaluate their own outputs and
    express appropriate epistemic humility.
    """
    print("\n" + "=" * 60)
    print("METACOGNITION DEMONSTRATION")
    print("=" * 60)

    memory = create_autonomous_memory()

    # Quality assessment
    print("\nQuality Assessment:")

    # High-quality response
    high_quality_response = (
        "Based on my analysis of the codebase, the authentication module "
        "uses JWT tokens for session management. The tokens are signed using "
        "RS256 and have a 24-hour expiration. This approach provides good "
        "security while maintaining stateless session handling."
    )

    assessment = memory.assess_quality(high_quality_response, confidence=0.92)
    print(f"\n  Response (high confidence):")
    print(f"    Quality Score: {assessment['quality_score']:.2f}")
    print(f"    Meets Quality Bar: {assessment['meets_quality_bar']}")
    print(f"    Recommendations: {assessment['recommendations'] or 'None'}")

    # Low-quality response
    low_quality_response = "It uses tokens."

    assessment = memory.assess_quality(low_quality_response, confidence=0.4)
    print(f"\n  Response (low confidence):")
    print(f"    Quality Score: {assessment['quality_score']:.2f}")
    print(f"    Meets Quality Bar: {assessment['meets_quality_bar']}")
    print(f"    Recommendations: {assessment['recommendations']}")

    # Uncertainty quantification
    print("\n\nUncertainty Quantification:")

    # Response with evidence
    response_with_evidence = (
        "The project deadline is likely next Friday, based on the team's "
        "typical sprint schedule and the manager's email."
    )

    uncertainty = memory.quantify_uncertainty(
        response_with_evidence,
        confidence=0.75,
        evidence=[
            "Email from project manager mentioning Friday",
            "Sprint board showing end date",
            "Team calendar event"
        ]
    )

    print(f"\n  Response with evidence:")
    print(f"    Uncertainty Score: {uncertainty['uncertainty_score']:.2f}")
    print(f"    Evidence Strength: {uncertainty['evidence_strength']:.2f}")
    print(f"    Should Seek Clarification: {uncertainty['should_seek_clarification']}")
    print(f"    Knowledge Gaps: {uncertainty['knowledge_gaps'] or 'None identified'}")

    # Response without evidence
    response_without_evidence = (
        "I think the API might possibly support pagination, "
        "but I'm not entirely sure about the implementation."
    )

    uncertainty = memory.quantify_uncertainty(
        response_without_evidence,
        confidence=0.35,
        evidence=[]
    )

    print(f"\n  Response without evidence:")
    print(f"    Uncertainty Score: {uncertainty['uncertainty_score']:.2f}")
    print(f"    Evidence Strength: {uncertainty['evidence_strength']:.2f}")
    print(f"    Should Seek Clarification: {uncertainty['should_seek_clarification']}")
    print(f"    Uncertainty Indicators: {uncertainty['uncertainty_indicators']}")
    print(f"    Knowledge Gaps: {uncertainty['knowledge_gaps']}")


def demonstrate_langchain_integration():
    """
    Demonstrate full LangChain integration with autonomous features.

    Shows how to use autonomous features alongside standard
    LangChain conversation chains.
    """
    print("\n" + "=" * 60)
    print("LANGCHAIN INTEGRATION DEMONSTRATION")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("\nSkipping LangChain demo - OPENAI_API_KEY not set")
        return

    memory = create_autonomous_memory()

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    print("\nRunning conversation with autonomous features:")

    # Start a working memory session for this conversation
    with memory.with_working_memory("conversation-001") as session:
        print(f"  Working memory session: {session['session_id']}")

        # Track goal for this conversation
        goal = memory.track_goal("answer-user-question", [
            "Understand the question",
            "Retrieve relevant context",
            "Generate response",
            "Validate quality"
        ])

        # Step 1: Understand question
        question = "What are the benefits of using TypeScript over JavaScript?"
        memory.add_to_working_memory("conversation-001", f"Question: {question}")
        memory.complete_goal_step("answer-user-question")

        # Step 2: Context is automatically retrieved by memory
        memory.complete_goal_step("answer-user-question")

        # Step 3: Generate response
        print(f"\n  Question: {question}")
        response = conversation.run(question)
        print(f"  Response: {response[:200]}...")
        memory.complete_goal_step("answer-user-question")

        # Step 4: Validate quality
        assessment = memory.assess_quality(response, confidence=0.85)
        memory.complete_goal_step("answer-user-question")

        print(f"\n  Quality Assessment:")
        print(f"    Score: {assessment['quality_score']:.2f}")
        print(f"    Meets Bar: {assessment['meets_quality_bar']}")

        # Check goal status
        goal_status = memory.get_goal_status("answer-user-question")
        print(f"\n  Goal Status: {goal_status['status']}")
        print(f"  Progress: {goal_status['progress']*100:.0f}%")


def show_autonomous_status():
    """Display the current status of autonomous features."""
    print("\n" + "=" * 60)
    print("AUTONOMOUS STATUS")
    print("=" * 60)

    memory = create_autonomous_memory()

    # Create some sessions and goals for demo
    memory.create_working_memory_session("demo-session-1")
    memory.track_goal("demo-goal-1", ["step1", "step2"])

    status = memory.get_autonomous_status()

    print(f"\n  Autonomous Enabled: {status['enabled']}")
    print(f"  Configuration:")
    for key, value in status['config'].items():
        print(f"    {key}: {value}")
    print(f"  Active Working Memory Sessions: {status['working_memory_sessions']}")
    print(f"    Sessions: {status['active_sessions']}")
    print(f"  Active Goals: {status['active_goals']}")
    for goal_id, goal_info in status['goals'].items():
        print(f"    {goal_id}: {goal_info['status']} ({goal_info['progress']*100:.0f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("RecallBricks LangChain v1.3.0 - Autonomous Agent Features")
    print("=" * 60)

    # Check for required API key
    if not os.getenv("RECALLBRICKS_API_KEY"):
        print("\nNote: RECALLBRICKS_API_KEY not set")
        print("Some features require a valid API key.")
        print("Get your key at: https://recallbricks.com")

    try:
        # Run all demonstrations
        demonstrate_working_memory()
        demonstrate_goal_tracking()
        demonstrate_metacognition()
        demonstrate_langchain_integration()
        show_autonomous_status()

        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("  - Working memory sessions with context manager")
        print("  - Goal tracking with progress monitoring")
        print("  - Quality assessment for metacognition")
        print("  - Uncertainty quantification with evidence")
        print("  - Decorator pattern for goal tracking")
        print("\nLearn more at: https://recallbricks.com/docs")

    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()
