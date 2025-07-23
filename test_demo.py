#!/usr/bin/env python3
"""Test the orchestrator demo option 1"""

import os
import sys

# Set PYTHONPATH
sys.path.insert(0, '/workspace')

# Test imports
try:
    from agent.orchestrator.demo import demo_full_development_workflow, create_orchestrator, IMPORTS_AVAILABLE
    print(f"✅ Imports successful (IMPORTS_AVAILABLE={IMPORTS_AVAILABLE})")
    
    # Run demo 1
    print("\nRunning demo_full_development_workflow()...")
    demo_full_development_workflow()
    
    print("\n✅ Demo completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()