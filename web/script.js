document.addEventListener('DOMContentLoaded', function() {
    fetch('/api/accidents') // You may need to update this endpoint to fetch specific data for road accidents
    .then(response => response.json())
    .then(data => {
        const accidentData = document.getElementById('accidentData');
        data.forEach((accident, index) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${index + 1}</td>
                <td>${accident.coordinates}</td>
                <td>${accident.time}</td>
            `;
            accidentData.appendChild(tr);
        });
    })
    .catch(error => console.error('Error:', error));
});
