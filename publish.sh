#!/bin/bash
set -e

usage() {
    echo "Usage: ./publish.sh [--major | --minor | --patch]"
    echo "  --major   Bump major version (X.0.0)"
    echo "  --minor   Bump minor version (x.Y.0)"
    echo "  --patch  Bump patch version (x.y.Z)"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

CURRENT_VERSION=$(sed -n 's/.*version="\([^"]*\)".*/\1/p' setup.py)
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

case "$1" in
    --major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    --minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    --patch)
        PATCH=$((PATCH + 1))
        ;;
    *)
        usage
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
TAG="v$NEW_VERSION"

echo "Current version: $CURRENT_VERSION"
echo "New version: $NEW_VERSION"
echo ""
read -p "Proceed with release $TAG? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

sed -i.bak "s/version=\"[^\"]*\"/version=\"$NEW_VERSION\"/" setup.py && rm setup.py.bak

git add setup.py
git commit -m "Bump version to $TAG"

git tag "$TAG"
git push origin main
git push origin "$TAG"

gh release create "$TAG" --generate-notes

echo "Released $TAG successfully!"
