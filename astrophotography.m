% Low-cost Computational Astrophotography
% Joseph Yen (EE 367/368), Peter Bryan (EE 367)
% 15 March 2019

% Run this file to get fully post-processed image
% Click on celestial pole in image and press ENTER to continue

clear all; close all;
n = 1; % Change input image # here
tic
I_color = im2double(imread(strcat('Data/',int2str(n),'.jpg')));
I_gray = rgb2gray(I_color); [y,x,z] = size(I_color); I_combined = zeros(y,x);
sw = 1; star_thresh = 0.9; pw = 5; patch_thresh = 0.8;

% Global Thresholding
I_globalbin = zeros(y,x,z); global_threshold = 0.5;
for i = sw+1:y-sw
    for j = sw+1:x-sw
        if I_gray(i,j) > global_threshold && I_globalbin(i,j) == 0
            for a = i-sw:i+sw
                for b = j-sw:j+sw
                    if I_gray(a,b)/I_gray(i,j) > star_thresh % Width estimation
                        I_globalbin(a,b,:) = 1;
                    end
                end
            end
        end
    end
end
I_global_binary = rgb2gray(I_globalbin);
I_global_binary(I_global_binary ~= 0) = 1;
for i = pw+1:y-pw
    for j = pw+1:x-pw
        window = I_global_binary(i-pw:i+pw,j-pw:j+pw);
        if sum(sum(window))/(2*pw+1)^2 > patch_thresh % Patch removal
            I_globalbin(i-pw:i+pw,j-pw:j+pw,:) = 0;
        end
    end
end
I_global = I_globalbin.*I_color; I_dg = I_color-I_global;

% Thresholding by Otsu's method
I_bin = im2bw(I_gray,graythresh(I_gray)); I_binary = I_bin;
for i = sw+1:y-sw
    for j = sw+1:x-sw
        if I_bin(i,j) == 1 && I_binary(i,j) == 0
            for a = i-sw:i+sw
                for b = j-sw:j+sw
                    if I_gray(a,b)/I_gray(i,j) > star_thresh
                        I_binary(a,b) = 1;
                    end
                end
            end
        end
    end
end
I_otsu_binary = I_binary;
for i = pw+1:y-pw
    for j = pw+1:x-pw
        window = I_otsu_binary(i-pw:i+pw,j-pw:j+pw);
        if sum(sum(window))/(2*pw+1)^2 > patch_thresh
            I_binary(i-pw:i+pw,j-pw:j+pw) = 0;
        end
    end
end
I_otsu = I_color;
for k = 1:3
    I_otsu(:,:,k) = I_otsu(:,:,k).*I_binary;
end
I_do = I_color-I_otsu;

% Thresholding by HSV
I_hsv = rgb2hsv(I_color); I_HSVbin = zeros(y,x,z);
val = I_hsv(:,:,3); val_threshold = mean(mean(val))*2;
for i = sw+1:y-sw
    for j = sw+1:x-sw
        if val(i,j) > val_threshold
            for a = i-sw:i+sw
                for b = j-sw:j+sw
                    if I_gray(a,b)/I_gray(i,j) > star_thresh
                        I_HSVbin(a,b,:) = 1;
                    end
                end
            end
        end
    end
end
I_HSV_binary = rgb2gray(I_HSVbin);
I_HSV_binary(I_HSV_binary ~= 0) = 1;
for i = pw+1:y-pw
    for j = pw+1:x-pw
        window = I_HSV_binary(i-pw:i+pw,j-pw:j+pw);
        if sum(sum(window))/(2*pw+1)^2 > patch_thresh
            I_HSVbin(i-pw:i+pw,j-pw:j+pw,:) = 0;
        end
    end
end
I_HSV = I_HSVbin.*I_color; I_dhsv = I_color-I_HSV;

% Thresholding by combining three methods
zero_threshold = 0.1;
gray_bin = I_gray; gray_bin(gray_bin ~= 0) = 1;
test = rgb2gray(I_dg); test(test ~= 0) = 1;
if sum(sum(gray_bin - test))/(x*y) < zero_threshold
    I_combined = I_combined + I_globalbin;
end
test = rgb2gray(I_do); test(test ~= 0) = 1;
if sum(sum(gray_bin - test))/(x*y) < zero_threshold
    I_combined = I_combined + I_binary;
end
test = rgb2gray(I_dhsv); test(test ~= 0) = 1;
if sum(sum(gray_bin - test))/(x*y) < zero_threshold
    I_combined = I_combined + I_HSVbin;
end
I_combined(I_combined ~= 0) = 1;
I_combined_color = I_color .* I_combined; I_dc = I_color-I_combined_color;

% Plot thresholding results
% figure(1); subplot(2,2,1); imshow(I_global); title('Global threshold')
% figure(2); subplot(2,2,1); imshow(I_dg); title('Global Difference')
% figure(1); subplot(2,2,2); imshow(I_otsu); title('Otsu method')
% figure(2); subplot(2,2,2); imshow(I_do); title('Otsu Difference')
% figure(1); subplot(2,2,3); imshow(I_HSV); title('HSV threshold')
% figure(2); subplot(2,2,3); imshow(I_dhsv); title('HSV Difference')
% figure(1); subplot(2,2,4); imshow(I_combined_color); title('Combined methods')
% figure(2); subplot(2,2,4); imshow(I_dc); title('Combined Difference')
figure(1); imshow(I_combined_color); title('Combined methods')
figure(2); imshow(I_dc); title('Combined Difference')

% Filling holes by linear interpolation then 2D averaging
I_thresholded = I_combined_color; I_diff = I_dc;
I_filled = I_diff; I_filled_gray = rgb2gray(I_filled);
for i = 2:y
    first = 2;
    for j = 2:x
        if I_filled_gray(i,j) == 0 && I_filled_gray(i,j-1) ~= 0
            first = j;
        elseif I_filled_gray(i,j) ~= 0 && I_filled_gray(i,j-1) == 0
            for a = first:j-1
                I_filled(i,a,:) = (I_filled(i,first-1,:)+I_filled(i,j,:))/2;
            end
        end
    end
end
w = 5; I_thresholded_bin = rgb2gray(I_thresholded);
for i = w+1:y-w
    for j = w+1:x-w
        if I_thresholded_bin(i,j) ~= 0
            window = I_filled(i-w:i+w,j-w:j+w,:);
            windowbin = rgb2gray(window);
            I_filled(i,j,:) = sum(sum(window,1),2)/nnz(windowbin);
        end
    end
end
figure(3); imshow(I_filled); title('Holes filled')

% Transform to polar
figure(4); imshow(I_color); title('Original image')
[xc,yc] = getpts(figure(4)); xc = round(xc); yc = round(yc);
r = linspace(0,sqrt((x-xc).^2+(y-yc).^2),1500);
theta = linspace(-pi,pi,200); pGrid = zeros(1500,200,3);
for a = 1:y
    for b = 1:x
        R = sqrt((b-xc).^2+(a-yc).^2);
        [~,rind] = min(abs(r-R));
        [~,tind] = min(abs(theta - atan2(a-yc,b-xc)));
        pGrid(rind,tind,:) = I_thresholded(a,b,:);
    end
end
pComp = zeros(size(pGrid));

% Compress Stars and convert to Cartesian
for a = 1:3
    pComp(:,:,a) = compressStars(pGrid(:,:,a));
end
I_stars = zeros(size(I_filled));
for a = 1:size(pComp,1)
    for b = 1:size(pComp,2)
        i = round(r(a).*cos(theta(b)))+xc+1;
        j = round(r(a).*sin(theta(b)))+yc+1;
        if(i<=size(I_stars,2) && j<=size(I_stars,1))
            if(i>0 && j>0)
                I_stars(j,i,:) = I_stars(j,i,:)+pComp(a,b,:);
            end
        end
    end
end

% Shape stars
if(x*y>250000)
    k = fspecial('gaussian',[3,3],1);
else
    k = fspecial('gaussian',[3,3],0.5);
end
k = k.*(1/(k(2:2)));
for c = 1:3
    temp = conv2(I_stars(:,:,c),k);
    I_stars(:,:,c) = temp(2:end-1,2:end-1);
end
pw = 4;
for i = pw+1:y-pw
    for j = pw+1:x-pw
        window = I_stars(i-pw:i+pw,j-pw:j+pw);
        if nnz(window) > 9
            I_stars(i-pw:i+pw,j-pw:j+pw,:) = 0;
        end
    end
end
I_tot = I_filled + I_stars;
figure(5); imshow(I_tot); title('Star modification')

% Bilateral filtering
window = fix(sqrt(x*y*10^-5));
if window > 1
    sigma = 0.8; sigma_i = 0.8; im = I_tot;
    [X,Y] = meshgrid(-window:window,-window:window);
    im_gaussian = exp(-(X.^2+Y.^2)/(2*sigma^2));
    im_bilateral = I_tot;
    im_padded = zeros(size(im,1)+window*2,size(im,2)+window*2,3);
    [num_row,num_col,c] = size(im_padded);
    im_padded(window+1:num_row-window,window+1:num_col-window,:) = im;
    for k = 1:3
        for i = window+1:num_row-window
            for j = window+1:num_col-window
                patch = im_padded(i-window:i+window,j-window:j+window,k);
                weight_i = exp((patch-im(i-window,j- window,k)).^2/(2*sigma_i^2));
                combined = weight_i.*im_gaussian(1:window*2+1,1:window*2+1);
                im_bilateral(i-window,j-window,k) = sum(sum(combined.*patch))/sum(combined(:));
            end
        end
    end
    figure(6); imshow(im_bilateral); title('Bilateral filtered')
    imwrite(im_bilateral,strcat(int2str(n),'_bilateral.jpg'))
end
toc

% Evaluate variance of a window
win0 = I_color(yc-15:yc+15,xc-15:xc+15,:);
win0r = reshape(rgb2gray(win0),[1,961]);
v0 = var(win0r)
win1 = I_diff(yc-15:yc+15,xc-15:xc+15,:);
win1r = reshape(rgb2gray(win1),[1,961]);
win1r = win1r(win1r~=0);
v1 = var(win1r)
win2 = I_filled(yc-15:yc+15,xc-15:xc+15,:);
win2r = reshape(rgb2gray(win2),[1,961]);
v2 = var(win2r)
window = 3; sigma = 1; sigma_i = 1; im = I_filled; 
[X,Y] = meshgrid(-window:window,-window:window);
im_gaussian0 = exp(-(X.^2+Y.^2)/(2*sigma^2));
im_bilateral = I_filled;
im_padded0 = zeros(size(im,1)+window*2,size(im,2)+window*2,3); [num_row,num_col,c] = size(im_padded0); im_padded0(window+1:num_row-window,window+1:num_col-window,:) = im; for k = 1:3
    for i = window+1:num_row-window
        for j = window+1:num_col-window
            patch = im_padded0(i-window:i+window,j-window:j+window,k);
            weight_i = exp((patch-im(i-window,j- window,k)).^2/(2*sigma_i^2));
            combined = weight_i.*im_gaussian0(1:window*2+1,1:window*2+1);
            im_bilateral(i-window,j-window,k) = sum(sum(combined.*patch))/sum(combined(:));
        end
    end
end
win3 = im_bilateral(yc-15:yc+15,xc-15:xc+15,:);
win3r = reshape(rgb2gray(win3),[1,961]);
v3 = var(win3r)