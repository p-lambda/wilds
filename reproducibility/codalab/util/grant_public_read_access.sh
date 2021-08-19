#
# Run this script to give the public read access to Wilds CodaLab worksheets and bundles.
#
# Usage: bash reproducibility/codalab/util/grant_public_read_access.sh
#
for i in $(cl wsearch 0x63397d8cb2fc463c80707b149c2d90d1 .limit=10 -u); do
    # cl wperm ${i} public read || true;
    cl perm $(cl search .mine host_worksheet=${i} .limit=1000 -u) public read || true;
done
