"""
examples.mock_demo
~~~~~~~~~~~~~~~~~~

A completely self-contained, offline demonstration of the Vision Bridge. 
Uses a simulated NumPy camera test pattern feed and a Mock VLM Client. 
Demonstrates both Iterator (Generators) and Observer (Callbacks) patterns 
simultaneously, running out of the box with zero hardware or API key dependencies.
"""

import sys
import os
import time

# Ensure parent package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision_bridge import VisionBridgeModule, VisionEvent


def simulated_heavy_reasoning_llm(context_description: str) -> str:
    """
    Simulates a heavy core reasoning LLM (e.g. GPT-4 or Llama 3 8B) 
    determining operational procedures based on visual cues.
    """
    cd_lower = context_description.lower()
    
    if "none" in cd_lower:
        return "Log: Heartbeat normal. No security or inventory actions required."
        
    if "technician" in cd_lower or "person" in cd_lower:
        return (
            "ACTION DETERMINED [SECURITY SYSTEM]:\n"
            "   -> 1. Verify safety badge scanning on terminal.\n"
            "   -> 2. Log entrance event in administrative ledger.\n"
            "   -> 3. Confirm technician is wearing safety gear (eyewear/gloves)."
        )
        
    if "wrench" in cd_lower or "tool" in cd_lower:
        return (
            "ACTION DETERMINED [INVENTORY SYSTEM]:\n"
            "   -> 1. Register tool state transition: WRENCH in use.\n"
            "   -> 2. Start checkout time log (Max checkout limit: 60 minutes).\n"
            "   -> 3. Update plant floor dashboard display."
        )
        
    return f"Log: Routine change observed: '{context_description}'. Action logged."


# Callback observer definition
def mock_telemetry_logger(event: VisionEvent) -> None:
    """A background logger observer tracking operational metrics."""
    print(f"\n📈 [METRIC LOG] Telemetry captured for frame #{event.frame_id}:")
    print(f"    Inference Latency: {event.response_time_ms:.1f}ms")
    print(f"    Source Channel    : {event.metadata.get('camera_index')}")
    print(f"    VLM Class         : {event.metadata.get('provider')}")


def main():
    print("======================================================================")
    print("        VISION BRIDGE MODULE - OFFLINE ZERO-SETUP DEMO")
    print("======================================================================\n")
    print("[Demo]: Starting module with simulated camera and mock VLM client...")

    # Initialize with mock model and mock camera feed index
    # We set frame_rate to 1.0 (1 analysis per second)
    vision_layer = VisionBridgeModule(
        vlm_model="mock",
        camera_index="mock",
        frame_rate=1.0,
        simulated_latency=0.25  # simulate API network delays
    )

    # Register our telemetry observer callback (Observer Pattern)
    vision_layer.add_callback(mock_telemetry_logger)

    # Prompt demanding specific keyword filtering
    filter_prompt = (
        "Look for objects or people. If a specific tool, person, or significant "
        "change is visible, state it clearly in 5 words or less. Otherwise reply 'None'."
    )

    # Start the engine
    vision_layer.start(prompt=filter_prompt)

    try:
        print("\n[Demo]: Consuming visual events via Iterator Pattern (Generator).")
        print("[Demo]: Main thread block will now read the stream.")
        print("[Demo]: Press Ctrl+C to terminate the demo.\n")

        # Run for 8 iterations or until interrupted
        iteration_limit = 8
        iteration_count = 0

        for event in vision_layer.get_events():
            iteration_count += 1
            if iteration_count > iteration_limit:
                print("\n[Demo]: Iteration limit reached. Initiating clean exit.")
                break

            visual_update = event.raw_text
            
            print(f"\n----------------------------------------------------------------------")
            print(f"👀 [Stream Event] Raw VLM Output: '{visual_update}'")
            print(f"----------------------------------------------------------------------")

            # Emulate filter check
            if "none" in visual_update.lower():
                print(" -> Status: Idle (Filtering out 'None' event to save heavy API tokens).")
                continue

            print(" -> Status: Significant Trigger Detected!")
            print(" -> Consulting simulated heavy reasoning LLM for system payload...")
            
            verdict = simulated_heavy_reasoning_llm(visual_update)
            print(f"\n{verdict}")
            print(f"----------------------------------------------------------------------")

            # Small sleep to keep print layouts readable
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[Demo]: Keyboard interrupt detected.")
    finally:
        # Stop core and clean up resources
        print("\n[Demo]: Releasing and stopping VisionBridge...")
        vision_layer.remove_callback(mock_telemetry_logger)
        vision_layer.stop()
        print("\n======================================================================")
        print("                 DEMO COMPLETED SUCCESSFULLY")
        print("======================================================================")


if __name__ == "__main__":
    main()
