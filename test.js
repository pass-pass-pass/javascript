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

function forEach(list, callback){
    for (var n = 0; n < list.length; n++){
     callback.call(list[n],n, list[n+ 1]);
    }
}
var numbers = [5,3,2,6];
forEach(numbers, function(index, index2){
    numbers[index]= this + index2;});
console.log(numbers);




function set(num, callback) {
    for (var i = 0; i < num.length; i++) {
        callback.call(num[i], i, i+ 1);
    }
}
set(numbers, function(index, index3) {
    numbers[index] = index + index3;
    }    );
console.log(numbers)


function Llama() {
    this.spitted = false;
    this.spit = function() { this.spitted = true; }
}
var s = new Llama;

console.log(s.spitted);
s.spit();
console.log(s.spitted);


function constructor() {
    this.a = 3;
    this.b = function() {
        this.a = 4;
    }
}

var c = new constructor();
console.log(c.a);
c.b();
console.log(c.a);
