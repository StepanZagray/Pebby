"""Run one LS20 operation from a JSON file (or stdin), locally or through HostAI.

The request is exactly what the browser sends: a single object with an `op`.
See `examples/` for one file per operation.
"""

import argparse
import json
import os
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

from inference import DEFAULT_CHECKPOINT, Engine

MODEL_NAME = "pebby:latest"
# A gateway is not necessarily friendly. Stop reading rather than following a
# stream that never sends its `done` record.
MAX_RECORDS = 1000
MAX_RESPONSE = 64 * 1024 * 1024


def validate_request(request):
    """The one check worth doing before a network round trip."""
    if not isinstance(request, dict) or not isinstance(request.get("op"), str):
        raise ValueError('Request must be a JSON object with a string "op".')
    return request


def hostai_predict(request, origin, token=None):
    validate_request(request)
    body = {"model": MODEL_NAME, "input": request}
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        path = "/guest/v1/infer"
    else:
        path = "/api/infer"
    req = Request(origin.rstrip("/") + path, data=json.dumps(body).encode(), headers=headers)
    result = None
    read = records = 0
    with urlopen(req, timeout=60) as response:
        for line in response:
            read += len(line)
            records += 1
            if read > MAX_RESPONSE or records > MAX_RECORDS:
                raise ValueError("HostAI response exceeded the size limit.")
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Each HostAI record must be a JSON object.")
            if record.get("error"):
                raise ValueError(record["error"])
            if "event" in record:
                result = record["event"]
            if record.get("done") is True:
                if result is None:
                    raise ValueError("HostAI response finished without a result.")
                return result
    raise ValueError("HostAI response ended without a done record.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", default="-", help="JSON file, or - for stdin")
    parser.add_argument("--checkpoint", default=None,
                        help=f"Trained LS20 policy for {{'op': 'agent'}}. Default: {DEFAULT_CHECKPOINT}")
    parser.add_argument("--hostai", help="HostAI owner or local guest origin, e.g. http://127.0.0.1:8080")
    parser.add_argument("--guest", action="store_true", help="Use local guest API with HOSTAI_ACCESS_TOKEN")
    parser.add_argument("--indent", type=int, default=None, help="Pretty-print the result")
    args = parser.parse_args()
    try:
        if args.guest and (not args.hostai or not os.environ.get("HOSTAI_ACCESS_TOKEN")):
            raise ValueError("--guest requires --hostai and the HOSTAI_ACCESS_TOKEN environment variable.")
        if args.request == "-":
            request = json.load(sys.stdin)
        else:
            with open(args.request) as file:
                request = json.load(file)
        if args.hostai:
            result = hostai_predict(request, args.hostai,
                                    os.environ["HOSTAI_ACCESS_TOKEN"] if args.guest else None)
        else:
            result = Engine(args.checkpoint).dispatch(validate_request(request))
        print(json.dumps(result, indent=args.indent))
    except (ValueError, OSError, URLError) as error:
        parser.exit(1, f"Request failed: {error}\n")


if __name__ == "__main__":
    main()
