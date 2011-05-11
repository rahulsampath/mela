
function saveImgData(imgMat, fnamePrefix, hx, hy, hz)

[Ny, Nx, Nz] = size(imgMat);

imageVec = zeros(Nx*Ny*Nz,1);
for k = 0:(Nz - 1)
    for j = 0:(Ny - 1)
        for i = 0:(Nx - 1)
            imageVec((((k*Ny) + j)*Nx) + i + 1) = imgMat(j + 1, i + 1, k + 1);
        end
    end
end

clear imgMat;

for i = 1:(Nx*Ny*Nz)
    if (imageVec(i) < 0)
        imageVec(i) = 0;
    end
    if (imageVec(i) > 255)
        imageVec(i) = 255;
    end
    imageVec(i) = floor(imageVec(i));
end

writeAnalyzeImage(fnamePrefix, Nx, Ny, Nz, hx, hy, hz, imageVec);

