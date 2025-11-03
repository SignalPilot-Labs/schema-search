#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./publish.sh <version>"
    echo "Example: ./publish.sh v0.1.9"
    exit 1
fi

VERSION=$1
VERSION_NUMBER="${VERSION#v}"

echo "Publishing version $VERSION..."

sed -i.bak "s/version=\"[^\"]*\"/version=\"$VERSION_NUMBER\"/" setup.py && rm setup.py.bak

git add setup.py
git commit -m "Bump version to $VERSION"

git tag "$VERSION"
git push origin main
git push origin "$VERSION"

gh release create "$VERSION" --generate-notes

echo "Released $VERSION successfully!"
