"""Send realistic test traffic to the API so the monitoring dashboard has data to show.

    python scripts/generate_traffic.py                         # 200 requests, normal patients
    python scripts/generate_traffic.py --count 500 --rate 10   # 500 requests, 10 per second
    python scripts/generate_traffic.py --profile high-risk     # simulate a shift in patients
    python scripts/generate_traffic.py --url http://127.0.0.1:8000

Mix of requests: ~85% complete patients, ~10% with missing fields, ~5% invalid (rejected, HTTP 422).
The "high-risk" profile sends older patients with more risk factors -- watch the predicted
'Disease' share and the probability distribution move on the dashboard (simulated data drift).
Uses only the Python standard library.
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter


def patient(profile: str) -> dict:
    high = profile == "high-risk"
    return {
        "age": round(random.gauss(62 if high else 53, 8)),
        "sex": int(random.random() < (0.85 if high else 0.68)),
        "cp": random.choices([1, 2, 3, 4], weights=[1, 1, 1, 7] if high else [1, 2, 3, 4])[0],
        "trestbps": round(random.gauss(140 if high else 130, 17)),
        "chol": round(random.gauss(250, 50)),
        "fbs": int(random.random() < 0.15),
        "restecg": random.choice([0, 1, 2]),
        "thalach": round(random.gauss(125 if high else 150, 20)),
        "exang": int(random.random() < (0.65 if high else 0.33)),
        "oldpeak": round(abs(random.gauss(2.0 if high else 0.9, 1.0)), 1),
        "slope": random.choice([1, 2, 3]),
        "ca": random.choices([0, 1, 2, 3], weights=[2, 3, 3, 2] if high else [6, 2, 1, 1])[0],
        "thal": random.choices([3, 6, 7], weights=[2, 1, 7] if high else [6, 1, 3])[0],
    }


def make_request(profile: str) -> dict:
    roll = random.random()
    body = patient(profile)
    if roll < 0.05:                                    # invalid input -> should get HTTP 422
        body["age"] = random.choice([250, -3, "unknown"])
    elif roll < 0.15:                                  # missing values -> pipeline imputes them
        for field in random.sample(["chol", "trestbps", "ca", "thal", "slope", "fbs"], k=2):
            body[field] = None
    return body


def post(url: str, body: dict) -> int:
    req = urllib.request.Request(f"{url}/predict", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code
    except urllib.error.URLError:
        return 0                                       # API not reachable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=200, help="number of requests")
    parser.add_argument("--rate", type=float, default=5, help="requests per second")
    parser.add_argument("--profile", choices=["normal", "high-risk"], default="normal")
    args = parser.parse_args()

    print(f"Sending {args.count} requests to {args.url} ({args.profile} patients, ~{args.rate}/s)...")
    statuses = Counter()
    for i in range(1, args.count + 1):
        statuses[post(args.url, make_request(args.profile))] += 1
        if statuses[0] >= 5 and statuses[0] == i:
            raise SystemExit(f"API not reachable at {args.url} -- is it running?")
        if i % 50 == 0:
            print(f"  {i} sent  {dict(statuses)}")
        time.sleep(1 / args.rate)
    print(f"Done. Responses by HTTP status: {dict(statuses)}")


if __name__ == "__main__":
    main()
