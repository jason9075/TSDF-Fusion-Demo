default:
    @just --list

dev:
    find . -not -path './.git/*' \( -name '*.html' -o -name '*.js' -o -name '*.css' \) \
        | entr -r python3 -m http.server 8080
