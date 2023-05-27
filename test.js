var classes = new Array();
classes[1] = 3;
var num = [13,3,4]
console.log(num.length);
num.push(4);
console.log(num.length);
num.sort()
console.log(num);
llama = {color: "brown", age:7, hasFur: true};
var sum =0;
for (var i =0; i < num.length; i++) {
    sum += num[i];
}
console.log(sum);

function eat() {
    return this;
}

var sleep = function () {
    return this;
}

sleep;
console.log(eat);
