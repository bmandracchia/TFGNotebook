% Leer la imagen original desde una ruta específica
imagen_original = imread('C:\Users\Yi\TFGNotebook\ImagenPrueba.jpg');

% Especificar la ruta completa para guardar la nueva imagen .tif
output_path = 'C:\Users\Yi\TFGNotebook\nueva_imagenPruebaPSF.tif';

% Guardar la imagen en formato .tif
imwrite(imagen_original, output_path);
