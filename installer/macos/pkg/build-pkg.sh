#!/usr/bin/env bash
# Build (and, where credentials exist, sign + notarize + staple) the macOS .pkg (§4.4).
#
#   build-pkg.sh <bundle-dir> <version> <output.pkg>
#
# Signing is CONDITIONAL and says which it did. Without a Developer ID the package still
# builds and installs — Installer.app warns about an unidentified developer and the user can
# proceed, which the tarball never let them do at all — and with one it is signed, notarized
# and stapled, so it opens with no warning and works offline (stapling is what makes the
# notarization ticket available without a network round-trip at install time).
#
# Deliberately NOT removing install.sh's quarantine strip yet. The plan says it "can go" after
# signing; it can go once signing is actually ON, and it is a no-op on a signed install anyway.
# Removing it while the release is still unsigned would take away the only thing standing
# between a marketer and "the developer cannot be verified".
set -euo pipefail

BUNDLE="${1:?usage: build-pkg.sh <bundle-dir> <version> <output.pkg>}"
VERSION="${2:?version}"
OUTPUT="${3:?output .pkg path}"

IDENTIFIER="com.campaignintelligence.app"
STAGING="/usr/local/share/campaign-intelligence-staging"
here="$(cd "$(dirname "$0")" && pwd)"

[ -x "$BUNDLE/campaign-intelligence" ] || {
    echo "no campaign-intelligence binary in $BUNDLE" >&2; exit 1; }
[ -f "$BUNDLE/install.sh" ] || {
    echo "$BUNDLE has no install.sh — the postinstall runs it, and a package whose payload" >&2
    echo "omits it installs nothing while reporting success." >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
root="$work/root$STAGING"
mkdir -p "$root"
cp -a "$BUNDLE"/. "$root"/
chmod +x "$root/install.sh" "$root/campaign-intelligence"

# Sign the binary FIRST, inside the payload. A signed package around an unsigned binary
# notarizes and then fails on launch — the .pkg's own signature says nothing about what is
# in it, and Gatekeeper checks the thing that runs.
if [ -n "${MACOS_SIGN_IDENTITY:-}" ]; then
    echo "Signing the binary as $MACOS_SIGN_IDENTITY ..."
    # --options runtime is required for notarization; --timestamp so the signature outlives
    # the certificate.
    #
    # Found by WHAT THE FILE IS, not by its extension. This matched `*.so` and `*.dylib`, and
    # a PyInstaller bundle carries Mach-O executables with no extension at all — `collect_all`
    # brings torch's `bin/protoc`, `bin/torch_shm_manager` and friends in as DATA, and a
    # framework build contributes `Python.framework/Versions/3.x/Python`. Notarization refuses
    # a payload containing an unsigned Mach-O, and a hardened main binary refuses to load an
    # unsigned library, so the extension list would have failed the release either late or
    # silently.
    while IFS= read -r target; do
        codesign --force --timestamp --options runtime \
                 --sign "$MACOS_SIGN_IDENTITY" "$target"
    done < <(find "$root" -type f ! -path "*/install.sh" ! -path "*/uninstall.sh" -exec sh -c \
                 'file -b "$1" | grep -q "Mach-O"' _ {} \; -print \
             | grep -v "/campaign-intelligence$")
    codesign --force --timestamp --options runtime \
             --sign "$MACOS_SIGN_IDENTITY" "$root/campaign-intelligence"
    codesign --verify --strict --verbose=2 "$root/campaign-intelligence"
else
    echo "No MACOS_SIGN_IDENTITY: building an UNSIGNED package."
fi

component="$work/app.pkg"
pkgbuild --root "$work/root" \
         --identifier "$IDENTIFIER" \
         --version "$VERSION" \
         --scripts "$here/scripts" \
         --install-location / \
         "$component"

if [ -n "${MACOS_SIGN_IDENTITY_INSTALLER:-}" ]; then
    productbuild --distribution "$here/distribution.xml" --package-path "$work" \
                 --sign "$MACOS_SIGN_IDENTITY_INSTALLER" "$OUTPUT"
else
    productbuild --distribution "$here/distribution.xml" --package-path "$work" "$OUTPUT"
fi

# Notarize + staple, when Apple credentials exist. Stapling is the half that matters offline:
# without it the first launch asks Apple whether the package is notarized, and the network
# this product is installed on may well not allow that (§4.2's whole position).
if [ -n "${AC_NOTARY_PROFILE:-}" ]; then
    echo "Notarizing ..."
    xcrun notarytool submit "$OUTPUT" --keychain-profile "$AC_NOTARY_PROFILE" --wait
    xcrun stapler staple "$OUTPUT"
    xcrun stapler validate "$OUTPUT"
    echo "Signed, notarized and stapled."
elif [ -n "${MACOS_SIGN_IDENTITY_INSTALLER:-}" ]; then
    echo "Signed but NOT notarized (no AC_NOTARY_PROFILE)." >&2
fi

pkgutil --check-signature "$OUTPUT" || true
echo "Built $OUTPUT"
