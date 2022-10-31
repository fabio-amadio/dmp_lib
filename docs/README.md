## Generate documentation ##

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

## References ##

* [Documenting Python code with Sphinx](https://towardsdatascience.com/documenting-python-code-with-sphinx-554e1d6c4f6d)
*[Generate professional python SDK documentation using sphinx](https://dock2learn.com/tech/generate-professional-python-sdk-documentation-using-sphinx/)
*[How to Host Your Sphinx Documentation on GitHub](https://python.plainenglish.io/how-to-host-your-sphinx-documentation-on-github-550254f325ae)
