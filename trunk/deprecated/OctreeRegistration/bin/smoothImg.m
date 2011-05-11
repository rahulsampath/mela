
function smoothImg(fnamePrefix, filterWidth)

scalingFactor = ((2*filterWidth) + 1)^3;

imgInfo = analyze75info(fnamePrefix);
img = double(analyze75read(imgInfo));

[Ny, Nx, Nz] = size(img);

if length(imgInfo.PixelDimensions)
   hx = double(imgInfo.PixelDimensions(1));
   hy = double(imgInfo.PixelDimensions(2));
   hz = double(imgInfo.PixelDimensions(3));
else
   hx = 0;
   hy = 0;
   hz = 0;
end

imgS = zeros(Ny, Nx, Nz);

for k = 1:Nz
    for i = 1:Nx
        for j = 1:Ny
            res = 0;
            for m = (-filterWidth):1:filterWidth
                for n = (-filterWidth):1:filterWidth
                    for p = (-filterWidth):1:filterWidth
                        iidx = i + m;
                        jidx = j + n;
                        kidx = k + p;
                        if( (jidx > 0) && (jidx <= Ny) ...
                                && (iidx > 0) && (iidx <= Nx) ...
                                && (kidx > 0) && (kidx <= Nz) )
                            res = res + img(jidx, iidx, kidx);
                        end
                    end
                end
            end
            imgS(j, i, k) = res/scalingFactor;
        end
    end
end

fnameNewPrefix = [fnamePrefix,'_Smooth'];

saveImgData(imgS, fnameNewPrefix, hx, hy, hz);

display(['New image stored in: ',fnameNewPrefix])





