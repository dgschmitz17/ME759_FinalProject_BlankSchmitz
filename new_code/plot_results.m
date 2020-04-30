push = csvread('processed_push.csv');
release = csvread('processed_release.csv');

figure
plot(push);
plot(release);