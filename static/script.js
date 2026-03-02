async function checkSpam() {
    const msg = document.getElementById("message").value;

    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: msg })
    });

    const data = await response.json();

    document.getElementById("result").innerText = data.result;
}