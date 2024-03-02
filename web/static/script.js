const canvas = document.getElementById("myCanvas");
const result = document.getElementById("result");
const ctx = canvas.getContext("2d");

let isDrawing = false;
let lastX = 0;
let lastY = 0;

function drawLine(x, y) {
    ctx.lineWidth = 15;
    ctx.strokeStyle = "rgba(0, 0, 0, 15)";
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.strokeStyle = "black";
    ctx.beginPath();
    ctx.moveTo(lastX + 8, lastY + 8);
    ctx.lineTo(x + 8, y + 8);
    ctx.stroke();
    lastX = x;
    lastY = y;
}

canvas.addEventListener("mousedown", (e) => {
    isDrawing = true;
    lastX = e.offsetX;
    lastY = e.offsetY;
});

canvas.addEventListener("mouseup", () => {
    isDrawing = false;
});

canvas.addEventListener("mousemove", (e) => {
    if (isDrawing) {
        drawLine(e.offsetX, e.offsetY);
    }
});

const sendButton = document.getElementById("send-btn");

sendButton.addEventListener("click", () => {
    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");
    tempCanvas.width = 28;
    tempCanvas.height = 28;

    tempCtx.drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);

    const data = tempCanvas.toDataURL("image/png");

    fetch("https://digitor.svs.gyarab.cz/upload_image", {
        method: "POST",
        body: JSON.stringify({
            input_data: data,
        }),
    })
        .then((response) => response.json())
        .then((data) => {
            console.log("Image upload response:", data);
            canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
            result.textContent = data["message"];
        })
        .catch((error) => {
            console.error("Error uploading image:", error);
        });
});