from __future__ import annotations

import argparse
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

SAMPLE_PATIENT = {
    "age": 60, "sex": 1, "cp": 1, "trestbps": 145, "chol": 133,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.4, "slope": 3, "ca": 0, "thal": 6,
}

# checks if the server is up using health api, takes server ip and max number of attempts
# then prepares the api request payload and calls the predict api
# prints the api repsonse
def test_run(base_url: str = "http://localhost:8000", attempts: int = 5) -> dict:
    for attempt in range(attempts):
        try:
            with urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    break
        except (URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(2)
    payload = json.dumps(SAMPLE_PATIENT).encode("utf-8")
    request = Request(
        f"{base_url}/predict", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urlopen(request, timeout=10) as response:
        result = json.load(response)
    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability_heart_disease"] <= 1
    print(json.dumps(result, indent=2))
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    test_run(args.url)

if __name__ == "__main__":
    main()
