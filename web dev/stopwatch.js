let startTime, updatedTime, difference;
let running = false;
let interval;

const display = document.getElementById('display');

function startStopwatch() {
    if (!running) {
        startTime = new Date().getTime() - (difference || 0);
        interval = setInterval(updateDisplay, 1000);
        running = true;
    }
}

function stopStopwatch() {
    clearInterval(interval);
    running = false;
}

function resetStopwatch() {
    clearInterval(interval);
    running = false;
    difference = 0;
    display.innerHTML = "00:00:00";
}

function updateDisplay() {
    updatedTime = new Date().getTime();
    difference = updatedTime - startTime;

    const hours = Math.floor((difference % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const minutes = Math.floor((difference % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((difference % (1000 * 60)) / 1000);

    display.innerHTML = (hours < 10 ? "0" : "") + hours + ":" + 
                        (minutes < 10 ? "0" : "") + minutes + ":" + 
                        (seconds < 10 ? "0" : "") + seconds;
}

// Event Listeners
document.getElementById('startButton').onclick = startStopwatch;
document.getElementById('stopButton').onclick = stopStopwatch;
document.getElementById('resetButton').onclick = resetStopwatch;
