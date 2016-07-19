function lol = persontest(args)
image = args.arg1;
load INRIA/inriaperson_final.mat;
im = imread(image);
[dets, boxes] = imgdetect(im, model, -0.3);
%disp(dets)
top = nms(dets, 0.5);
%disp(dets)
%disp(boxes(top,:));
%showboxes(im, reduceboxes(model, boxes(top,:)));
%disp(reduceboxes(model, boxes(top,:)))
lol = reduceboxes(model, boxes(top,:));
end