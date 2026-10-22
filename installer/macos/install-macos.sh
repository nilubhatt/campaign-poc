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
#   CAMPAIGN_INTELLIGENCE_VERSION=v0.5.0   install that release instead of the latest
#
# The whole body is a function called on the last line, so a download cut short by the
# network runs nothing at all rather than half of it.
set -euo pipefail

main() {
  local repo="nilubhatt/campaign-poc"
  local asset="campaign-intelligence-macos-arm64.tar.gz"
  # Where the release's files are. The override is how this script is tested against a
  # locally built bundle before it is ever published.
  local base="${CAMPAIGN_INTELLIGENCE_RELEASE_URL:-}"
  if [ -z "$base" ]; then
    if [ -n "${CAMPAIGN_INTELLIGENCE_VERSION:-}" ]; then
      base="https://github.com/$repo/releases/download/$CAMPAIGN_INTELLIGENCE_VERSION"
    else
      base="https://github.com/$repo/releases/latest/download"
    fi
  fi

  if [ "$(uname -s)" != "Darwin" ]; then
    fail "This installer is for macOS. Linux and Windows have their own: see https://github.com/$repo#installation"
  fi
  # The macOS bundle is built for Apple silicon only.
  if [ "$(uname -m)" != "arm64" ]; then
    fail "This Mac has an $(uname -m) processor, and Campaign Intelligence for macOS needs Apple silicon (M1 or later)."
  fi
  # About 800 MB to download, 1.3 GB unpacked, and the installed copy beside it.
  local free_kb
  free_kb="$(df -Pk "${TMPDIR:-/tmp}" | awk 'NR==2 {print $4}')"
  if [ -n "$free_kb" ] && [ "$free_kb" -lt 4000000 ]; then
    fail "Not enough free disk space: about 4 GB is needed while installing."
  fi

  local work
  work="$(mktemp -d "${TMPDIR:-/tmp}/campaign-intelligence.XXXXXX")"
  # shellcheck disable=SC2064  # expanded now, on purpose: $work is local to this function
  trap "rm -rf '$work'" EXIT

  echo "Downloading Campaign Intelligence (about 800 MB)..."
  curl -fL --retry 3 --progress-bar -o "$work/$asset" "$base/$asset" \
    || fail "The download failed. Check the internet connection and run this again."
  curl -fsSL --retry 3 -o "$work/$asset.sha256" "$base/$asset.sha256" \
    || fail "Could not download the checksum that proves the download is complete. Run this again."

  echo "Checking the download..."
  # The checksum file names the archive, so it is checked by name, in its own directory.
  ( cd "$work" && shasum -a 256 -c "$asset.sha256" >/dev/null ) \
    || fail "The download is damaged or incomplete (its checksum does not match). Run this again."

  echo "Unpacking..."
  tar -xzf "$work/$asset" -C "$work"
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
