1. Inside the __root__ folder, run:
```
sphinx-apidoc -o docs dmp_lib/
```
2. Inside the __docs/__ folder, run:
```
make html
```
3. Move the whole contents of folder ___build/html/__ inside of the __docs/__ folder.
Then push everything in the __gh-pages__ branch.
