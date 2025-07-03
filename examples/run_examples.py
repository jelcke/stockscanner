#!/usr/bin/env python3
"""
Interactive Brokers Stock Scanner Examples Runner
Select and run different scanner examples
"""

import subprocess
import sys
from pathlib import Path


def print_menu():
    """Print the examples menu"""
    print("\n" + "="*60)
    print("Interactive Brokers Stock Scanner Examples")
    print("="*60)
    print("\n1. Basic Scanner - Real IB connection with delayed data")
    print("2. Demo Scanner - Simulated data (no IB required)")
    print("3. Advanced Display - Rich terminal interface demo")
    print("4. Test IB Connection - Verify your IB setup")
    print("5. Pattern Detection Demo - ML pattern recognition")
    print("6. Sentiment Analysis Demo - News sentiment analysis")
    print("7. Exit")
    print("\n" + "-"*60)

def run_example(script_name):
    """Run the selected example script"""
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"Error: {script_name} not found")
        return

    print(f"\nRunning {script_name}...")
    print("Press Ctrl+C to stop\n")

    try:
        subprocess.run([sys.executable, str(script_path)])
    except KeyboardInterrupt:
        print("\n\nExample stopped by user")
    except Exception as e:
        print(f"Error running example: {e}")

def main():
    """Main menu loop"""
    while True:
        print_menu()

        try:
            choice = input("\nSelect an example (1-7): ").strip()

            if choice == '1':
                print("\n⚠️  This requires IB Gateway/TWS running on port 7497")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    run_example("basic_scanner.py")

            elif choice == '2':
                run_example("demo_scanner.py")

            elif choice == '3':
                run_example("advanced_display.py")

            elif choice == '4':
                print("\n⚠️  This requires IB Gateway/TWS running")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    run_example("test_ib_connection.py")

            elif choice == '5':
                run_example("pattern_detection.py")

            elif choice == '6':
                run_example("sentiment_integration.py")

            elif choice == '7':
                print("\nGoodbye!")
                break

            else:
                print("\nInvalid choice. Please select 1-7.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
