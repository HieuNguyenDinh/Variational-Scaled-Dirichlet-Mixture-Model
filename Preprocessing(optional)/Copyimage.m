% close;
% clear;
% clc
image_folder='D:/ThesisCode/Fan Code/test/';
destination_folder = 'D:/ThesisCode/Fan Code/test-temp/';
% filename = readtable('test.csv');
filenames = table2struct(test);
% filenames=dir(fullfile(image_folder,'*.ppm'));
total_images=numel(filenames);

for i = 1:total_images
%     f=f ullfile(image_folder, filenames(i).name);
    f = strcat(image_folder, filenames(i).name);
    f1 = convertStringsToChars(f);
    if isfile(f1)
        copyfile(f1, destination_folder);
%         movefile(f1, destination_folder);
%         delete(f1)
    end
end