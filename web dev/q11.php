<?php

/*if ($_SERVER["REQUEST_METHOD"] == "GET") {
    $name =htmlspecialchars($_GET["name"]);
    $email = htmlspecialchars($_GET["email"]);
    echo "<h3>Data Received Using GET</h3>";
    echo "Name: $name<br>";
    echo "Email: $email<br>";
}*/
 
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $name =htmlspecialchars($_POST["name"]);
    $email = htmlspecialchars($_POST["email"]);
    echo "<h3>Data Received Using POST</h3>";
    echo "Name: $name<br>";
    echo "Email: $email<br>";
 }
?>

