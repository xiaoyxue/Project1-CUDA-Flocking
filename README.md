**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* (TODO) YOUR NAME HERE
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

### Running on Windows

Generate and build the Visual Studio 2026 x64 project:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
```

Open `build\cis5650_boids.slnx` in Visual Studio, or run from PowerShell:

```powershell
Set-Location .\build\bin\Release
.\cis5650_boids.exe
```

CMake copies `shaders` beside the executable and sets Visual Studio's debugger
working directory to that output directory. Shader paths are relative to the
working directory; when launching from a terminal, use the executable's directory
as shown above. Keep the `shaders` directory with the executable when moving it.
