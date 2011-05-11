
function saveMismatches(imgFixed, imgMoving, imgResult, slices, prefixStr)

colorScale = 256;

[N, dummy, dummy] = size(imgFixed);

initMismatch = abs(imgFixed - imgMoving);
finalMismatch = abs(imgFixed - imgResult);

scaleFac = max(max(max(initMismatch)));

initMismatch = (colorScale-1)*(initMismatch/scaleFac);
finalMismatch = (colorScale-1)*(finalMismatch/scaleFac);

for i = slices               
    axis off;    
    fhandle = figure(1);
    image(initMismatch(:,:,i));
    title('Initial Mismatch')
    colormap(gray(colorScale));
    fname = [prefixStr,'InitSlice', int2str(i), '.jpg'];
    saveas(fhandle, fname);
    
    axis off;    
    fhandle = figure(2);
    image(finalMismatch(:,:,i));
    title('Final Mismatch')
    colormap(gray(colorScale));        
    fname = [prefixStr,'FinalSlice', int2str(i), '.jpg'];
    saveas(fhandle, fname);
end

