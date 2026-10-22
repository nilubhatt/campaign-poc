#!/usr/bin/env bash
# Campaign Intelligence on a Mac, in one command. Nothing to copy and paste:
#
#   curl -fsSL https://github.com/nilubhatt/campaign-poc/releases/latest/download/install-macos.sh | bash
#
# or download this file and run `bash install-macos.sh`. It downloads the macOS bundle from
# the release, checks it against the SHA-256 published beside it, unpacks it and runs the
# bundle's own install.sh, which verifies the weights, uses the embedder that ships inside
# the bundle (nothing is fetched from ollama.com), runs the self-test and only then connects
# Claude Desktop. Running it again upgrades in place. Arguments are passed to install.sh.
#
# A specific release instead of the latest (the variable goes on `bash`, not on `curl`):
#
#   curl -fsSL https://github.com/nilubhatt/campaign-poc/releases/latest/download/install-macos.sh | CAMPAIGN_INTELLIGENCE_VERSION=v0.5.0 bash
#
# The whole body is a function called on the last line, so a download cut short by the
# network runs nothing at all rather than half of it.
set -euo pipefail

main() {
  local repo="nilubhatt/campaign-poc"
  local asset="campaign-intelligence-macos-arm64.tar.gz"
  # HTTPS only, redirects included. The override (how this script is tested against a locally
  # built bundle before it is published) may be plain http on this machine.
  local proto=(--proto '=https' --proto-redir '=https')
  local base="${CAMPAIGN_INTELLIGENCE_RELEASE_URL:-}"
  if [ -n "$base" ]; then
    proto=()
  elif [ -n "${CAMPAIGN_INTELLIGENCE_VERSION:-}" ]; then
    base="https://github.com/$repo/releases/download/$CAMPAIGN_INTELLIGENCE_VERSION"
  else
    base="https://github.com/$repo/releases/latest/download"
  fi

  if [ "$(uname -s)" != "Darwin" ]; then
    fail "This installer is for macOS. Linux and Windows have their own: see https://github.com/$repo#installation"
  fi
  # The macOS bundle is built for Apple silicon only. Asked of the hardware, not of the
  # shell: a terminal running under Rosetta reports x86_64 on an M-series Mac.
  if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" != "1" ] && [ "$(uname -m)" != "arm64" ]; then
    fail "This Mac has an Intel processor, and Campaign Intelligence for macOS needs Apple silicon (M1 or later)."
  fi
  # About 800 MB to download and 1.3 GB unpacked here, and the 1.3 GB installed copy in the
  # home folder; on most Macs both are the same disk, and the smaller figure is what counts.
  local free_kb
  # `|| true`: df exits non-zero when one of the paths does not exist yet, and under pipefail
  # that ended the script with no word said.
  free_kb="$( { df -Pk "${TMPDIR:-/tmp}" "${HOME:-/}" 2>/dev/null || true; } | awk 'NR>1 {print $4}' | sort -n | head -1)"
  if [ -n "$free_kb" ] && [ "$free_kb" -lt 4000000 ]; then
    fail "Not enough free disk space: about 4 GB is needed while installing."
  fi

  local work
  work="$(mktemp -d "${TMPDIR:-/tmp}/campaign-intelligence.XXXXXX")"
  # shellcheck disable=SC2064  # expanded now, on purpose: $work is local to this function
  trap "rm -rf '$work'" EXIT

  # The checksum first: a few bytes that say whether this release can be installed at all,
  # before 800 MB is spent finding out.
  if ! curl -fsSL --retry 3 ${proto[@]+"${proto[@]}"} -o "$work/$asset.sha256" "$base/$asset.sha256"; then
    if [ -n "${CAMPAIGN_INTELLIGENCE_VERSION:-}" ]; then
      fail "There is no release $CAMPAIGN_INTELLIGENCE_VERSION with a macOS installer. Release names look like v0.5.0; see https://github.com/$repo/releases"
    fi
    fail "Could not download the release's checksum, so the download could not be checked. Check the internet connection and run this again."
  fi
  local expected
  expected="$(awk 'NR==1 {print tolower($1)}' "$work/$asset.sha256" | tr -d '\r')"
  case "$expected" in
    *[!0-9a-f]*|"") fail "The release's checksum file is not a SHA-256. Run this again later." ;;
  esac
  [ "${#expected}" -eq 64 ] || fail "The release's checksum file is not a SHA-256. Run this again later."

  echo "Downloading Campaign Intelligence (about 800 MB)..."
  curl -fL --retry 3 --progress-bar ${proto[@]+"${proto[@]}"} -o "$work/$asset" "$base/$asset" \
    || fail "The download failed. Check the internet connection and run this again."

  echo "Checking the download..."
  # The digest of the file just downloaded, compared with the published one: not `shasum -c`,
  # which checks whatever file the checksum line names.
  local actual
  actual="$(shasum -a 256 "$work/$asset" | awk '{print $1}')"
  [ "$actual" = "$expected" ] \
    || fail "The download is damaged or incomplete (its checksum does not match). Run this again."

  echo "Unpacking..."
  tar -xzf "$work/$asset" -C "$work" \
    || fail "The download could not be unpacked. Run this again."
  # Only a browser marks files as downloaded-from-the-internet, but a copy that arrived some
  # other way may carry the mark too; macOS then refuses the unsigned binary outright.
  xattr -dr com.apple.quarantine "$work/campaign-intelligence" 2>/dev/null || true

  # stdin from /dev/null: run as `curl ... | bash`, stdin is this script. With the body in a
  # function called on the last line, bash has read all of it by now; this keeps it that way
  # if a line is ever added after the call.
  bash "$work/campaign-intelligence/install.sh" "$@" < /dev/null
}

fail() {
  echo >&2
  echo "Campaign Intelligence was not installed: $*" >&2
  exit 1
}

main "$@"
