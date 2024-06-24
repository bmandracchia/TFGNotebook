%%Funcion para ver el resultado en 3D

function verTIF(filename)
volume = tiffreadVolume(filename);
min_val = double(min(volume(:)));
max_val = double(max(volume(:)));
volume_rescaled = double(volume - min_val) / (max_val - min_val);
implay(volume_rescaled)
end