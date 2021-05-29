pyinstaller ^
    --onefile ^
    --noconsole ^
    --noconfirm ^
    --clean ^
    --hidden-import "skimage.filters.rank.core_cy_3d" ^
    --add-data "svm_data.dat;." ^
    --add-data "trained_MNIST_model.h5;." ^
    --add-data "th.pgm;." ^
    --add-data "skel.pgm;." ^
    --add-binary ".\venv\Lib\site-packages\pylsd\lib\win32\x64\liblsd.dll;." ^
    --name "Circuit-Solver" ^
    schematic_generator.py