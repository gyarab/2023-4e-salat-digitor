const canvas = document.getElementById("myCanvas");
const result = document.getElementById("result");
const ctx = canvas.getContext("2d");
ctx.lineWidth = 25;

let isDrawing = false;
let x = 0;
let y = 0;

canvas.addEventListener("mousedown", (e) => {
    isDrawing = true;
    x = Math.floor(e.offsetX);
    y = Math.floor(e.offsetY);
    ctx.beginPath();
    ctx.moveTo(x, y);
});

canvas.addEventListener("mouseup", () => {
    isDrawing = false;
    ctx.closePath();
});

canvas.addEventListener("mousemove", (e) => {
    if (isDrawing) {
        x = Math.floor(e.offsetX);
        y = Math.floor(e.offsetY);
        ctx.lineTo(x, y);
        ctx.stroke();
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