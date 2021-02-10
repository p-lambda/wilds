#
# Run this script to give the public read access to Wilds CodaLab worksheet and bundles.
# Usage: bash reproducibility/codalab/util/grant_public_read_access.sh
#
for i in $(cl wsearch wilds-results .limit=1 -u); do
    # cl wperm ${i} public read || true;
    cl perm $(cl search .mine host_worksheet=${i} .limit=1000 -u) public read || true;
done
