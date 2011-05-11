
function saveMismatches(imgFixed, imgMoving, imgResult, slices, prefixStr)

colorScale = 256;

[N, dummy, dummy] = size(imgFixed);

for i = slices               
    axis off;    
    fhandle = figure(1);
    image(imgFixed(:,:,i));
    title('Fixed')
    colormap(gray(colorScale));
    fname = [prefixStr,'FixedSlice', int2str(i), '.jpg'];
    saveas(fhandle, fname);
    
    axis off;    
    fhandle = figure(2);
    image(imgMoving(:,:,i));
    title('Moving')
    colormap(gray(colorScale));        
    fname = [prefixStr,'MovingSlice', int2str(i), '.jpg'];
    saveas(fhandle, fname);
    
    axis off;    
    fhandle = figure(3);
    image(imgResult(:,:,i));
    title('Registered')
    colormap(gray(colorScale));
    fname = [prefixStr,'ResultSlice', int2str(i), '.jpg'];
    saveas(fhandle, fname);    
end

