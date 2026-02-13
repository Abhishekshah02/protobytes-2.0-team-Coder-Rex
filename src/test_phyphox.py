"""
Test if we can read data from Phyphox
Run this AFTER starting Phyphox remote access on phone
"""

import requests
import time
import json

# Change this to YOUR phone's IP (shown in Phyphox)
PHYPHOX_IP = "http://192.168.4.227:8080"

def test_connection():
    """Test basic connection to Phyphox"""
    
    print(f"Connecting to Phyphox at: {PHYPHOX_IP}")
    
    try:
        response = requests.get(f"{PHYPHOX_IP}/get?accX&accY&accZ", timeout=5)
        data = response.json()
        print(f"✅ Connected! Response: {json.dumps(data, indent=2)[:500]}")
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


def read_live_data(duration=5):
    """Read live data for specified seconds"""
    
    print(f"\nReading data for {duration} seconds...")
    print("Make sure Phyphox is PLAYING (▶ button)")
    
    all_x = []
    all_y = []
    all_z = []
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            response = requests.get(
                f"{PHYPHOX_IP}/get?accX&accY&accZ", 
                timeout=2
            )
            data = response.json()
            
            # Extract values
            if 'buffer' in data:
                x_vals = data['buffer']['accX']['buffer']
                y_vals = data['buffer']['accY']['buffer']
                z_vals = data['buffer']['accZ']['buffer']
                
                all_x.extend(x_vals)
                all_y.extend(y_vals)
                all_z.extend(z_vals)
                
                print(f"  Got {len(x_vals)} new readings (Total: {len(all_x)})")
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(0.5)
    
    print(f"\nTotal readings collected: {len(all_x)}")
    print(f"Sample X values: {all_x[:5]}")
    print(f"Sample Y values: {all_y[:5]}")
    print(f"Sample Z values: {all_z[:5]}")
    
    return all_x, all_y, all_z


if __name__ == "__main__":
    # First change the IP below to YOUR phone's IP
    PHYPHOX_IP = input("Enter Phyphox IP (e.g., http://192.168.1.5:8080): ").strip()
    
    if test_connection():
        print("\n✅ Connection successful!")
        print("\nNow press PLAY ▶ in Phyphox and start moving...")
        input("Press Enter when ready...")
        
        x, y, z = read_live_data(duration=5)
        
        if len(x) > 128:
            print(f"\n✅ Got enough data for prediction! ({len(x)} readings)")
        else:
            print(f"\n⚠️ Need more data. Got {len(x)}, need 128+")