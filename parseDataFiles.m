function [ data ] = parseDataFiles( dataFile, fileLabel, convert2Lower )
%UNTITLED2 Summary of this function goes here
%   fileLabel is the Label of all the data in dataFile (i.e - positive = 1/negative = -1)
    file = fopen(dataFile);
    data = {};
    line = fgetl(file);
    while ischar(line)
        [~, stripedLine] = stripLine(line);
        if strcmp(stripedLine, 'review')
            review = struct;
            review.label = fileLabel;
            review = parseInnerText(stripedLine, file, review);
            if convert2Lower
                review.review_text = lower(review_text);
            end
            %split to words and remove punctuation marks
            review.review_words = strsplit(review.review_text(~isstrprop(review.review_text, 'punct'))); 
            data = [data review];
        end
        line = fgetl(file);
    end
end

function [ review ] = parseInnerText(tag, file, review)
    value = '';
    line = fgetl(file);
    [isTag, stripedLine] = stripLine(line);
    while ischar(line) && ~strcmp(stripedLine, tag) %check if this is the end tag of the open "tag"
        if isTag %this is a new opening tag
            review = parseInnerText(stripedLine, file, review);
        else
            value = [value ' ' stripedLine];
        end
        line = fgetl(file);
        [isTag, stripedLine] = stripLine(line);
    end
    review.(char(tag)) = strtrim(value);
end

%checks if a line is a tag line.
%return the original line if not, and the tag if it is. 
function [isTag, lineTag] = stripLine(line)
    lineTag = line;
    isTag = false;
    reg = '<\/?([a-zA-Z_]*)>';
    toks = regexp(line, reg, 'tokens');
    if ~isempty(toks)
        isTag = true;
        lineTag = toks{1};
    end
end

