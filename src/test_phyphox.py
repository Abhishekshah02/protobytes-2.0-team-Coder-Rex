"""
Fixed Phyphox Data Collector - Gets 128+ readings
"""

import requests
import time
import json
import csv

PHYPHOX_IP = "192.168.4.227:8080"  # <-- CHANGE THIS TO YOUR PHONE'S IP ADDRESS


def test_connection():
    print(f"Connecting to Phyphox at: {PHYPHOX_IP}")
    try:
        response = requests.get(f"{PHYPHOX_IP}/get?accX=full", timeout=5)
        data = response.json()
        print("✅ Connected!")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect! Check:")
        print("   1. Phyphox 'Allow Remote Access' is ON")
        print("   2. Phone and laptop on SAME WiFi")
        print(f"   3. IP address is correct: {PHYPHOX_IP}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def collect_data(duration=10):
    """
    Step 1: Clear old data
    Step 2: Let Phyphox record for 'duration' seconds
    Step 3: Fetch FULL buffer at once
    """

    # --- CLEAR OLD DATA ---
    print("\n🗑️  Clearing old data...")
    try:
        requests.get(f"{PHYPHOX_IP}/control?cmd=clear", timeout=3)
        time.sleep(0.5)
    except:
        print("⚠️  Could not clear, continuing anyway...")

    # --- START RECORDING ---
    print("▶️  Starting Phyphox recording...")
    try:
        requests.get(f"{PHYPHOX_IP}/control?cmd=start", timeout=3)
    except:
        print("⚠️  Could not auto-start. Make sure PLAY is pressed!")

    # --- WAIT FOR DATA TO BUILD UP ---
    print(f"⏳ Recording for {duration} seconds...")
    print("   (Move phone / walk / keep still - whatever you need)\n")

    for i in range(duration):
        remaining = duration - i
        print(f"   ⏱️  {remaining} seconds remaining...")
        time.sleep(1)

    # --- FETCH FULL BUFFER AT ONCE ---
    print("\n📥 Fetching ALL data from buffer...")

    try:
        response = requests.get(
            f"{PHYPHOX_IP}/get?accX=full&accY=full&accZ=full&acc_time=full", timeout=10
        )
        data = response.json()

        x_vals = data["buffer"]["accX"]["buffer"]
        y_vals = data["buffer"]["accY"]["buffer"]
        z_vals = data["buffer"]["accZ"]["buffer"]

        # Try to get time, might have different name
        try:
            t_vals = data["buffer"]["acc_time"]["buffer"]
        except:
            t_vals = list(range(len(x_vals)))
            print("   (No timestamp buffer, using index)")

        # --- REMOVE None VALUES ---
        clean_x, clean_y, clean_z, clean_t = [], [], [], []

        for i in range(len(x_vals)):
            if (
                x_vals[i] is not None
                and y_vals[i] is not None
                and z_vals[i] is not None
            ):
                clean_x.append(x_vals[i])
                clean_y.append(y_vals[i])
                clean_z.append(z_vals[i])
                clean_t.append(t_vals[i] if t_vals[i] is not None else i)

        return clean_t, clean_x, clean_y, clean_z

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("\n🔧 Try this: Check what buffers exist...")
        try:
            r = requests.get(f"{PHYPHOX_IP}/get?accX=full", timeout=5)
            print(json.dumps(r.json(), indent=2)[:500])
        except:
            pass
        return [], [], [], []


def save_to_csv(t, x, y, z, filename="accelerometer_data.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z"])
        for i in range(len(x)):
            writer.writerow([t[i], x[i], y[i], z[i]])
    print(f"💾 Saved to {filename}")


if __name__ == "__main__":
    PHYPHOX_IP = input("Enter Phyphox IP (e.g., http://192.168.1.5:8080): ").strip()

    if not test_connection():
        exit()

    print("\n" + "=" * 50)
    print("📱 Make sure you have 'Accelerometer' open in Phyphox")
    print("=" * 50)
    input("\nPress ENTER to start collecting...")

    # --- COLLECT FOR 10 SECONDS ---
    t, x, y, z = collect_data(duration=10)

    print(f"\n{'=' * 50}")
    print(f"📊 RESULTS")
    print(f"{'=' * 50}")
    print(f"Total clean readings: {len(x)}")

    if len(x) >= 128:
        print(f"🎉 SUCCESS! Got {len(x)} samples (needed 128)")

        # Show stats
        print(f"\nFirst 5 readings:")
        for i in range(min(5, len(x))):
            print(f"  X={x[i]:.4f}  Y={y[i]:.4f}  Z={z[i]:.4f}")

        if len(t) > 1 and isinstance(t[0], float):
            total_time = t[-1] - t[0]
            hz = len(x) / total_time if total_time > 0 else 0
            print(f"\n📈 Duration: {total_time:.2f} seconds")
            print(f"📈 Sampling rate: {hz:.1f} Hz")

        # Save
        save_to_csv(t, x, y, z)
        print(f"\n✅ You're ready to use this data for prediction!")

    else:
        print(f"❌ Only got {len(x)} readings, need 128")
        print("\n🔧 FIXES TO TRY:")
        print("   1. Increase duration to 15 or 20 seconds")
        print("   2. Make sure Phyphox experiment is 'Accelerometer'")
        print("   3. Check if buffer names are different:")

        # Debug: show available buffer names
        try:
            r = requests.get(f"{PHYPHOX_IP}/get?accX=full", timeout=5)
            print(f"\n   Available data: {json.dumps(r.json(), indent=2)[:300]}")
        except:
            pass
