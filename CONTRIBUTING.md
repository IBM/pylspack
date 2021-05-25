## Contributing In General
This project was developed as part of the experimental evaluation of the algorithms
described in the respective publication. ***There is no plan for further development***. Nevertheless, if you would like to address an issue, please first
contact the maintainers via e-mail. For a list of the maintainers, see the [MAINTAINERS.md](MAINTAINERS.md) page.

Contributions, if any, will be handled by first creating [an issue](https://github.com/IBM/pylspack/issues),
and then opening a respective [pull request](https://github.com/IBM/pylspack/pulls).

**Note: We appreciate your effort, and want to avoid a situation where a contribution
requires extensive rework (by you or by us), sits in backlog for a long time, or
cannot be accepted at all!**

## Legal

We have tried to make it as easy as possible to make contributions. This
applies to how we handle the legal aspects of contribution. We use the
same approach - the [Developer's Certificate of Origin 1.1 (DCO)](https://github.com/hyperledger/fabric/blob/master/docs/source/DCO1.1.txt) - that the Linux® Kernel [community](https://elinux.org/Developer_Certificate_Of_Origin)
uses to manage code contributions.

We simply ask that when submitting a patch for review, the developer
must include a sign-off statement in the commit message.

Here is an example Signed-off-by line, which indicates that the
submitter accepts the DCO:

```
Signed-off-by: John Doe <john.doe@example.com>
```

You can include this automatically when you commit a change to your
local git repository using the following command:

```
git commit -s
```

## Install

Requirements:
- C++11 capable compiler
- CMake >= 3.11
- OpenMP >= 4.0 (simd directives)
- Numpy and SciPy (ideally with OpenMP support)

The installation will build the C/C++ code to generate the shared library that is used by the python wrappers. ***NOTE***: In order to keep the code architecture-independent the compiler optimization flags are kept as generic as possible. In order to apply additional optimization flags, simply add them in the `${PYLSPACK_ADDITIONAL_CMAKE_CXX_FLAGS}` environment variable prior to executing pip install:
```bash
python3 -m venv venv
source venv/bin/activate
# Optional step to add more compiler flags:
# export PYLSPACK_ADDITIONAL_CMAKE_CXX_FLAGS="-march=native -g"
pip install .
```

## Testing

To run the tests:
```bash
# If you get an error about liblinalg_kernels.so, do the following:
# PYLSPACK_LOCATION="$(pip show pylspack | grep Location: | awk '{print $2}')/pylspack/)"
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYLSPACK_LOCATION}
pip install -r test_requirements.txt
pytest -svvv test
```

## Coding style guidelines
We use [yapf](https://github.com/google/yapf) to format python code and [astyle](http://astyle.sourceforge.net/) for C++.
Make sure that in the last commit you have executed the following commands such that the code will be appropriately formatted.
***NOTE***: `astyle` will pad an empty line between `#pragma omp simd / parallel for` directives. Unfortunatelly, these empty lines have to be removed manually!
```bash
yapf -i --style "{based_on_style: pep8, blank_line_before_nested_class_or_def: true, indent_dictionary_value: true, dedent_closing_brackets: true, column_limit: 99}" --recursive .
find . | grep "\.h" | xargs astyle --suffix=none --max-code-length=120 --indent=spaces=2 --pad-oper --convert-tabs --unpad-paren --delete-empty-lines
find . | grep "\.h" | xargs astyle --suffix=none --max-code-length=120 --indent=spaces=2 --add-braces --convert-tabs --align-pointer=name --style=google --indent-classes --pad-oper --pad-paren-in --pad-header --break-blocks --break-after-logical 
```
