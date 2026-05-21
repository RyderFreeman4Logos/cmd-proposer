set shell := ["bash", "-e", "-u", "-o", "pipefail", "-c"]

default: pre-commit

pre-commit: find-monolith-files check-generated-artifacts check-charset fmt deny lint test

pre-push: check-version-bumped

find-monolith-files:
    @scripts/hooks/find-monolith-files.sh

check-generated-artifacts:
    @scripts/hooks/check-generated-artifacts.sh

check-version-bumped:
    @scripts/hooks/check-version-bumped.sh

check-charset:
    @scripts/hooks/check-charset.sh

fmt:
    @scripts/hooks/format-staged-rust.sh

deny:
    @cargo deny -L error check

lint:
    @cargo clippy --workspace --all-targets -- --cap-lints warn

lint-strict:
    @cargo clippy --workspace --all-targets -- -D warnings

test:
    @cargo nextest run --workspace

install:
    @cargo install --path crates/cps --locked --force

install-hooks:
    @git config --unset core.hooksPath 2>/dev/null || true
    @lefthook install
    @echo "Hooks installed."

bump-patch:
    @cargo set-version --workspace --bump patch
    @cargo metadata --format-version 1 >/dev/null
