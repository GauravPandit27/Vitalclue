"""
VitalCue - Model download helper

Pretrained weights are fetched on first run rather than committed, so a fresh clone stays
small. Getting the bytes down is the awkward part on managed machines, and this module
exists because a single urllib call is not enough. On the corporate laptop this was
developed on, all of the following happen against the same URL:

  urllib          CERTIFICATE_VERIFY_FAILED - the TLS-inspecting proxy's root sits in the
                  Windows certificate store, which Python does not consult; it ships its
                  own bundle instead.
  curl            CRYPT_E_NO_REVOCATION_CHECK - schannel can validate the proxy root but
                  cannot reach the revocation endpoint it names.
  curl --ssl-revoke-best-effort   works.

So the transports are tried in order, cheapest and most correct first, and every failure
is reported together if none succeed. Installing `truststore` (in requirements.txt) fixes
the first case properly by pointing Python at the OS trust store, which is why it is tried
before any of the subprocess fallbacks.

Downloads land in a .part file and are renamed only once complete, so an interrupted run
cannot leave a truncated model behind to fail later with a confusing deserialisation error.
"""
import os
import shutil
import subprocess
import sys
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Every model we fetch is far larger than this. Anything smaller is a proxy error page or
# a truncated transfer, both of which are more useful to catch here than at load time.
MIN_PLAUSIBLE_BYTES = 50_000


def _via_urllib(url: str, dest: str):
    urllib.request.urlretrieve(url, dest)


def _via_truststore(url: str, dest: str):
    """urllib, but validating against the OS trust store rather than Python's bundle."""
    import truststore  # noqa: PLC0415 - optional dependency, absence is a normal outcome

    truststore.inject_into_ssl()
    urllib.request.urlretrieve(url, dest)


def _curl(url: str, dest: str, extra=()):
    exe = shutil.which("curl") or shutil.which("curl.exe")
    if not exe:
        raise RuntimeError("curl not found")
    subprocess.run([exe, "-fsSL", "--retry", "2", *extra, url, "-o", dest],
                   check=True, capture_output=True)


def _via_curl(url: str, dest: str):
    _curl(url, dest)


def _via_curl_lenient_revocation(url: str, dest: str):
    """Windows only: schannel fails closed when it cannot reach a revocation endpoint."""
    if sys.platform != "win32":
        raise RuntimeError("not applicable off Windows")
    _curl(url, dest, ["--ssl-revoke-best-effort"])


def _via_powershell(url: str, dest: str):
    """Invoke-WebRequest uses .NET, which reads the Windows certificate store directly."""
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if not exe:
        raise RuntimeError("powershell not found")
    subprocess.run(
        [exe, "-NoProfile", "-NonInteractive", "-Command",
         f"Invoke-WebRequest -Uri '{url}' -OutFile '{dest}' -UseBasicParsing"],
        check=True, capture_output=True,
    )


TRANSPORTS = (
    _via_urllib,
    _via_truststore,
    _via_curl,
    _via_curl_lenient_revocation,
    _via_powershell,
)


def ensure_file(url: str, path: str, label: str) -> str:
    """Return `path`, downloading it first if absent. Raises if every transport fails."""
    if os.path.exists(path) and os.path.getsize(path) >= MIN_PLAUSIBLE_BYTES:
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = path + ".part"
    print(f"Downloading {label} -> {path}")

    failures = []
    for transport in TRANSPORTS:
        try:
            transport(url, partial)
            size = os.path.getsize(partial)
            if size < MIN_PLAUSIBLE_BYTES:
                raise RuntimeError(f"got only {size} bytes")
            os.replace(partial, path)
            return path
        except Exception as exc:  # noqa: BLE001 - exhaust every transport before failing
            detail = getattr(exc, "stderr", b"") or b""
            suffix = f" ({detail.decode(errors='replace').strip()})" if detail else ""
            failures.append(f"{transport.__name__}: {type(exc).__name__}: {exc}{suffix}")
        finally:
            if os.path.exists(partial):
                os.remove(partial)

    raise RuntimeError(
        f"Could not download {label} from {url}\n  " + "\n  ".join(failures)
        + f"\nDownload it manually and save it as {path}"
    )
