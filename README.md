# stackerhour

> Homage to the "Stackerday" sample scene

This is not really a library, but rather a bad attempt to recreate the "stacking" aspects of the "Stackerday" scene I saw decades ago in POV-Ray.
I can't find any original image anymore, but there is a [forum thread](https://forums.oculusvr.com/de/discussion/comment/340083/) which uses it.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [TODOs](#todos)
- [NOTDOs](#notdos)
- [Contribute](#contribute)

## Install

You're not really supposed to use it *for* anything.
But if you want to play around with it, feel free to `pip install -r requirements.txt` and you should be ready to go.

## Usage

Just use it!  It's not an all-purpose renderer, and for example I skipped the pruning part,
where elements outside the field of view are skipped.
This leads to crazy artifacts if things are *behind* your camera.

## TODOs

* Fix intersections.  Apparently I haven't fixed *all* causes yet.
* Make a nice (possibly low-res) gif from `history/`.
* Make the boxes look more like letter blocks.

## NOTDOs

Here are some things this project will definitely not support:
* General-purpose rendering.
* Anything non-squary.
* Advanced projection techniques.  Feel free to add isomorphic and other basic techniques though!

## Contribute

Feel free to dive in! [Open an issue](https://github.com/BenWiederhake/stackerhour/issues/new) or submit PRs.
