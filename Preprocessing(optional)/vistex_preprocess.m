close;
clear;
clc
image_folder='D:/ThesisCode/Fan Code/test';
filenames=dir(fullfile(image_folder,'*.ppm'));
% image_folder='D:/ThesisCode/Fan Code/dtd';
% filenames=dir(fullfile(image_folder,'*.jpg'));
total_images=numel(filenames);

for i = 1:total_images
    f=fullfile(image_folder, filenames(i).name);
    img = imread(f);
    img = rgb2gray(img);
    H1{i}= mat2tiles(img,ceil(size(img)/8));
end

data_num = total_images * 64;
H2 = zeros(data_num, 12);
temp = 1;

for h = 1:numel(H1)
    a = H1{h};
    for i = 1:8
        for j = 1:8
            glcm = graycomatrix(a{i,j},'offset',[0 1; -1 1; -1 0; -1 -1]);
            stats = graycoprops(glcm, {'Contrast','Energy','Homogeneity'});
            c = struct2cell(stats);
            concat_c = horzcat(c{1}, c{2}, c{3});
            H2(temp, :) = concat_c;
            temp = temp+1;
        end
    end
end

csvwrite('D:/Downloads BED/texture.csv', H2);  

            
            
            