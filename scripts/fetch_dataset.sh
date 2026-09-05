#!/usr/bin/env bash
# Pull the FLIR ADAS v2 download from Google Drive onto a rented box.
#
#   GDRIVE_ID=<file id> ./scripts/fetch_dataset.sh
#   ./scripts/fetch_dataset.sh 'https://drive.google.com/file/d/<id>/view?usp=sharing'
#
# FLIR puts the dataset behind a registration form, so it cannot be fetched
# from source on an unattended box. Keeping your own copy in Drive and pulling
# it from there is the practical answer -- this script is the other half of it.
#
# The file must be shared as "Anyone with the link". A Drive file that still
# requires sign-in returns an HTML login page, and gdown will cheerfully save
# that as a 4 KB "zip".
set -euo pipefail

DEST="${DEST:-Dataset}"
TARGET="${DEST}/FLIR_ADAS_v2"
ZIP="${DEST}/flir_adas_v2.zip"
KEEP_ZIP="${KEEP_ZIP:-0}"
# Anything smaller than this is a sign-in page or a quota notice, not the
# dataset. Overridable so the script can be exercised against a small archive.
MIN_ZIP_MB="${MIN_ZIP_MB:-100}"

GDRIVE_ID="${GDRIVE_ID:-}"
if [ $# -ge 1 ]; then
    # Accept a full share URL and pull the id out of either of the two shapes
    # Drive hands you: /file/d/<id>/view and ?id=<id>.
    GDRIVE_ID=$(printf '%s' "$1" | sed -nE 's#.*/file/d/([^/]+).*#\1#p; s#.*[?&]id=([^&]+).*#\1#p')
    [ -n "${GDRIVE_ID}" ] || GDRIVE_ID="$1"
fi

if [ -d "${TARGET}/images_thermal_train" ]; then
    echo "==> ${TARGET} already present; nothing to do."
    exit 0
fi
if [ -z "${GDRIVE_ID}" ]; then
    echo "Set GDRIVE_ID, or pass the Drive share URL as the first argument." >&2
    exit 1
fi

echo "==> Checking disk"
# The zip is ~13 GB and unpacks to ~13 GB, so extraction needs both at once.
# POSIX df, not `df -BG --output=avail`: the GNU spelling is unavailable on
# macOS and BusyBox, and a script that cannot be run locally does not get run
# locally before it is trusted with a paid instance.
AVAIL_GB=$(df -Pk . | awk 'NR==2 {print int($4/1048576)}')
echo "    ${AVAIL_GB} GB available"
if [ "${AVAIL_GB}" -lt 35 ]; then
    echo "    Under 35 GB free. The zip and the unpacked copy have to coexist" >&2
    echo "    during extraction. Re-rent with more disk rather than fighting this." >&2
    exit 1
fi

command -v unzip >/dev/null 2>&1 || {
    echo "==> Installing unzip"
    apt-get update -qq && apt-get install -y -qq unzip
}

mkdir -p "${DEST}"

if [ ! -f "${ZIP}" ]; then
    echo "==> Downloading from Google Drive (~13 GB)"
    # uvx rather than a project dependency: gdown is only ever needed here, and
    # adding it to pyproject would put it in the training image too.
    uvx gdown --no-cookies -O "${ZIP}" "https://drive.google.com/uc?id=${GDRIVE_ID}" || {
        cat >&2 <<'ERR'
    Download failed. The two usual causes:

    "Cannot retrieve the public link" -- the file is not shared as
        "Anyone with the link". Fix the sharing, or use rclone with your own
        credentials instead.

    "Too many users have viewed or downloaded this file recently" -- Drive's
        quota on public downloads. It clears in ~24h. The reliable workaround
        is to copy the file into your own Drive (right-click -> Make a copy)
        and use the new id, which resets the counter.
ERR
        exit 1
    }
fi

# A sign-in page or a quota notice saves as a small HTML file with a .zip name.
SIZE_MB=$(du -m "${ZIP}" | cut -f1)
if [ "${SIZE_MB}" -lt "${MIN_ZIP_MB}" ]; then
    echo "    ${ZIP} is only ${SIZE_MB} MB -- that is an error page, not the dataset:" >&2
    head -c 400 "${ZIP}" >&2
    echo >&2
    rm -f "${ZIP}"
    exit 1
fi
echo "    got ${SIZE_MB} MB"

echo "==> Extracting"
STAGING="${DEST}/.unpack"
rm -rf "${STAGING}"
mkdir -p "${STAGING}"
unzip -q "${ZIP}" -d "${STAGING}"

# The zip may hold the six directories at its root, or nested under a folder
# with any name. Find whichever level actually contains them.
ANCHOR=$(find "${STAGING}" -maxdepth 3 -type d -name images_thermal_train -print -quit)
if [ -z "${ANCHOR}" ]; then
    echo "    No images_thermal_train/ anywhere in the archive. Contents:" >&2
    find "${STAGING}" -maxdepth 2 | head -20 >&2
    exit 1
fi
mv "$(dirname "${ANCHOR}")" "${TARGET}"
rm -rf "${STAGING}"

echo "==> Verifying"
MISSING=0
for d in images_thermal_train images_thermal_val video_thermal_test \
         images_rgb_train images_rgb_val video_rgb_test; do
    if [ -f "${TARGET}/${d}/coco.json" ]; then
        printf "    %-22s %6s frames\n" "${d}" "$(ls "${TARGET}/${d}/data" | wc -l | tr -d ' ')"
    else
        echo "    ${d}: MISSING coco.json" >&2
        MISSING=1
    fi
done
[ -d "${TARGET}/images_thermal_train/analyticsData" ] \
    || { echo "    analyticsData/ missing -- the 16-bit arms cannot be built." >&2; MISSING=1; }
[ "${MISSING}" -eq 0 ] || exit 1

if [ "${KEEP_ZIP}" = "0" ]; then
    echo "==> Removing the zip (KEEP_ZIP=1 to keep it)"
    rm -f "${ZIP}"
fi

echo "==> Dataset ready at ${TARGET}"
